import copy
import os
import random

import numpy as np
import torch


def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model = model.module if hasattr(model, 'module') else model
    torch.save(model.state_dict(), os.path.join(model_dir, 'pytorch_model.bin'))


def restore_model(model, model_dir, device):
    state = torch.load(os.path.join(model_dir, 'pytorch_model.bin'), map_location=device)
    model.load_state_dict(state, strict=True)
    return model


class EarlyStopping:
    """Select the checkpoint with the highest dev accuracy.

    ``min_select_epoch`` restricts selection to epochs at or after the given
    one-based epoch. Early stopping itself is unaffected by this lower bound.
    """

    def __init__(self, patience, min_select_epoch=1, delta=1e-6):
        self.patience = int(patience)
        self.min_select_epoch = int(min_select_epoch)
        if self.min_select_epoch < 1:
            raise ValueError('min_select_epoch must be >= 1')
        self.delta = delta
        self.counter = 0
        self.best_score = 1e-6
        self.early_stop = False
        self.best_model = None
        self.best_head = None
        self.best_selected_score = float('-inf')
        self.best_selected_model = None
        self.best_selected_head = None
        self.selected_epoch = None

    def __call__(self, score, model, head=None, epoch=None):
        if score is None or not np.isfinite(score):
            raise ValueError(f'non-finite evaluation score: {score!r}')
        improved = score >= (self.best_score + self.delta)
        if improved:
            self.counter = 0
            self.best_score = score
            self.best_model = copy.deepcopy(model)
            self.best_head = copy.deepcopy(head)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        eligible = epoch is None or int(epoch) >= self.min_select_epoch
        if eligible and score >= (self.best_selected_score + self.delta):
            self.best_selected_score = score
            if epoch is not None:
                self.selected_epoch = int(epoch)
            if improved:
                self.best_selected_model = self.best_model
                self.best_selected_head = self.best_head
            else:
                self.best_selected_model = copy.deepcopy(model)
                self.best_selected_head = copy.deepcopy(head)
