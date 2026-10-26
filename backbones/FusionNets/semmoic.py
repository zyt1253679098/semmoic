import math

import torch
from torch import nn
from torch.nn.parameter import Parameter

from .semmoic_fusion import Fusion


class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=32, device=None):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.weight = Parameter(torch.Tensor(out_dims, in_dims).to(device))
        self.scale = scale
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        norm_x = torch.norm(input, 2, 1, keepdim=True)
        ex = input / norm_x
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(ex * self.scale, ew.t())


class MLP_head(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        if num_classes == 2:
            raise ValueError('The released model does not use the binary head')
        self.output_layer_1 = CosNorm_Classifier(
            args.base_dim, args.num_labels, args.scale, args.device)
        self.contrast_head = nn.Sequential(nn.Linear(args.base_dim, args.num_labels))

    def forward(self, x):
        return self.output_layer_1(x), self.contrast_head(x)


class MMEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = Fusion(args)

    def forward(self, text_feats, video_feats, audio_feats):
        return self.model(text_feats, video_feats, audio_feats)
