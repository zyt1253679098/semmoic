import torch
from torch import nn


class SupConLoss(nn.Module):
    """Supervised contrastive loss with explicit positive masks."""

    def __init__(self, contrast_mode='all'):
        super().__init__()
        self.contrast_mode = contrast_mode

    def forward(self, features, mask=None, temperature=0.07):
        if features.dim() < 3:
            raise ValueError('features must have shape [bsz, n_views, ...]')
        if features.dim() > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'unknown contrast mode: {self.contrast_mode}')

        anchor_dot_contrast = (anchor_feature @ contrast_feature.T) / temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        ignore_mask = torch.zeros_like(mask, dtype=torch.bool)
        ignore_mask[torch.eye(batch_size * anchor_count, dtype=torch.bool)] = 1
        mask = mask * (~ignore_mask)

        exp_logits = torch.exp(logits) * (~ignore_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()
