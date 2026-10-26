"""Stage 1: train the multimodal representation and export frozen features."""
import logging
import os
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from backbones.FusionNets.semmoic import MLP_head, MMEncoder
from data import MMDataset, build_dev, build_test, build_train, build_wild
from data.text import TextTokenizer
from losses import SupConLoss
from utils.functions import EarlyStopping, restore_model, save_model
from utils.metrics import AverageMeter


class SemMoic:
    def __init__(self, config, data_root, with_wild=True):
        self.config = config
        self.logger = logging.getLogger('SemMoic')
        self.device = torch.device(config.device)

        tokenizer = TextTokenizer(config.text_pretrained_model, config.seq_len['text'])
        train = build_train(config, data_root, tokenizer, include_anti=False)
        self.train_dataset = MMDataset(train)
        dev = MMDataset(build_dev(config, data_root, tokenizer))
        test = MMDataset(build_test(config, data_root, tokenizer))

        self.train_loader = self._loader(self.train_dataset, config.train_batch_size,
                                         shuffle=True, drop_last=True)
        self.dev_loader = self._loader(dev, config.eval_batch_size,
                                       shuffle=False, drop_last=False)
        self.test_loader = self._loader(test, config.eval_batch_size,
                                        shuffle=False, drop_last=False)
        self.wild_dataset = None
        self.aux_loader = None
        if with_wild:
            self.wild_dataset = build_wild(config, data_root, tokenizer,
                                           len(self.train_dataset))
            self.aux_loader = self._loader(self.wild_dataset, config.train_batch_size,
                                           shuffle=True, drop_last=True)

        self.mm_encoder = MMEncoder(config).to(self.device)
        self.classifier = MLP_head(config, config.num_labels).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.contrast_criterion = SupConLoss()

    def _loader(self, dataset, batch_size, shuffle, drop_last):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )

    def _set_optimizer(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params = list(self.mm_encoder.named_parameters()) + \
            list(self.classifier.named_parameters())
        grouped = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        optimizer = AdamW(grouped, lr=self.config.lr)
        n_train = self.config.num_train_examples
        steps = int(n_train / self.config.train_batch_size) * self.config.train_epochs
        warmup = int(n_train * self.config.train_epochs *
                     self.config.warmup_proportion / self.config.train_batch_size)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=steps)
        return optimizer, scheduler

    def _train_epoch(self, optimizer, scheduler, iter_aux, epoch):
        self.mm_encoder.train()
        self.classifier.train()
        loss_record = AverageMeter()
        ood_label = self.config.ood_label_id

        for batch in self.train_loader:
            text = batch['text_feats'].to(self.device)
            video = batch['video_feats'].to(self.device)
            audio = batch['audio_feats'].to(self.device)
            labels = batch['label_ids'].to(self.device)
            batch_size = labels.shape[0]

            aux_batch = next(iter_aux)
            text = torch.cat([text, aux_batch['text_feats'][:batch_size].to(self.device)], dim=0)
            video = torch.cat([video, aux_batch['video_feats'][:batch_size].to(self.device)], dim=0)
            audio = torch.cat([audio, aux_batch['audio_feats'][:batch_size].to(self.device)], dim=0)

            feats = self.mm_encoder(text, video, audio)
            mixed_labels = torch.cat(
                [labels, torch.full((batch_size,), ood_label, dtype=torch.long,
                                    device=self.device)], dim=0)
            logits, contrast_logits = self.classifier(feats)

            id_mask = mixed_labels < ood_label
            id_loss = self.criterion(logits[id_mask], mixed_labels[id_mask])

            feats_aug = self.mm_encoder(text, video, audio)
            _, contrast_logits_aug = self.classifier(feats_aug)
            contrast_size = feats_aug.shape[0]

            labels_expand = mixed_labels.expand(contrast_size, contrast_size)
            pos_mask = torch.eq(labels_expand, labels_expand.T)
            pos_mask[mixed_labels == ood_label, :] = 0
            pos_mask[torch.eye(contrast_size, dtype=torch.bool, device=self.device)] = 1

            contrastive_logits = torch.stack(
                [F.normalize(contrast_logits), F.normalize(contrast_logits_aug)], dim=1)
            contrastive_loss = self.contrast_criterion(
                contrastive_logits, mask=pos_mask, temperature=self.config.temperature)

            loss = id_loss + contrastive_loss
            optimizer.zero_grad()
            loss.backward()
            loss_record.update(loss.item(), batch_size)
            optimizer.step()
            scheduler.step()

        return loss_record.avg

    @torch.no_grad()
    def _get_outputs(self, loader):
        self.mm_encoder.eval()
        self.classifier.eval()
        labels, preds, feats = [], [], []
        for batch in loader:
            text = batch['text_feats'].to(self.device)
            video = batch['video_feats'].to(self.device)
            audio = batch['audio_feats'].to(self.device)
            batch_labels = batch['label_ids'].to(self.device)
            batch_feats = self.mm_encoder(text, video, audio)
            logits, _ = self.classifier(batch_feats)
            labels.append(batch_labels.cpu())
            preds.append(logits.argmax(dim=1).cpu())
            feats.append(batch_feats.cpu())
        return {
            'y_true': torch.cat(labels).numpy(),
            'y_pred': torch.cat(preds).numpy(),
            'y_feat': torch.cat(feats).numpy(),
        }

    def train(self):
        if self.wild_dataset is None:
            raise RuntimeError('representation training requires wild data')
        self.config.num_train_examples = len(self.train_dataset)
        optimizer, scheduler = self._set_optimizer()
        early_stopping = EarlyStopping(self.config.patience, self.config.min_select_epoch)
        iter_aux = None

        for epoch in range(1, self.config.train_epochs + 1):
            if (epoch - 1) % 5 == 0:
                self.wild_dataset.resample()
                iter_aux = cycle(self.aux_loader)
            loss = self._train_epoch(optimizer, scheduler, iter_aux, epoch)
            outputs = self._get_outputs(self.dev_loader)
            score = float((outputs['y_pred'] == outputs['y_true']).mean())
            self.logger.info('epoch %d train_loss=%.4f dev_acc=%.4f', epoch, loss, score)
            early_stopping(score, self.mm_encoder, self.classifier, epoch)
            if early_stopping.early_stop:
                self.logger.info('early stop at epoch %d', epoch)
                break

        if self.config.min_select_epoch > 1:
            if early_stopping.best_selected_model is None:
                raise RuntimeError(f'no eligible checkpoint at or after epoch {self.config.min_select_epoch}')
            self.mm_encoder = early_stopping.best_selected_model
            self.classifier = early_stopping.best_selected_head
        else:
            self.mm_encoder = early_stopping.best_model
            self.classifier = early_stopping.best_head
        self.logger.info('selected epoch=%s dev_acc=%.4f',
                         early_stopping.selected_epoch, early_stopping.best_selected_score)
        return early_stopping.selected_epoch

    @torch.no_grad()
    def export_features(self, data_root):
        tokenizer = TextTokenizer(self.config.text_pretrained_model,
                                  self.config.seq_len['text'])
        full_train = build_train(self.config, data_root, tokenizer, include_anti=True)
        full_loader = self._loader(MMDataset(full_train), self.config.eval_batch_size,
                                   shuffle=False, drop_last=False)

        train_out = self._collect(full_loader)
        test_out = self._get_outputs(self.test_loader)
        head = self.classifier.output_layer_1.weight.detach().cpu().numpy()

        return {
            'train_feat': train_out['y_feat'],
            'train_label': train_out['y_true'],
            'train_index': np.asarray(full_train.indices),
            'test_feat': test_out['y_feat'],
            'test_true': test_out['y_true'],
            'head_W': head,
        }

    @torch.no_grad()
    def _collect(self, loader):
        self.mm_encoder.eval()
        self.classifier.eval()
        labels, feats = [], []
        for batch in loader:
            text = batch['text_feats'].to(self.device)
            video = batch['video_feats'].to(self.device)
            audio = batch['audio_feats'].to(self.device)
            batch_labels = batch['label_ids'].to(self.device)
            batch_feats = self.mm_encoder(text, video, audio)
            labels.append(batch_labels.cpu())
            feats.append(batch_feats.cpu())
        return {
            'y_true': torch.cat(labels).numpy(),
            'y_feat': torch.cat(feats).numpy(),
        }

    def save(self, output_dir):
        save_model(self.mm_encoder, os.path.join(output_dir, 'mm_encoder'))
        save_model(self.classifier, os.path.join(output_dir, 'multiclass_classifier'))

    def restore(self, output_dir):
        restore_model(self.mm_encoder, os.path.join(output_dir, 'mm_encoder'), self.device)
        restore_model(self.classifier, os.path.join(output_dir, 'multiclass_classifier'),
                      self.device)

