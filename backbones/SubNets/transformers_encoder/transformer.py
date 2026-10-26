import math

import torch.nn.functional as F
from torch import nn

from .multihead_attention import MultiheadAttention
from .position_embedding import SinusoidalPositionalEmbedding


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0,
                 res_dropout=0.0, embed_dropout=0.0):
        super().__init__()
        self.dropout = embed_dropout
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(TransformerEncoderLayer(
                embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
            ))

        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in):
        x = self.embed_scale * x_in
        x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1,
                 res_dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x):
        residual = x
        x = self.self_attn(query=self.layer_norms[0](x))
        x = residual + F.dropout(x, p=self.res_dropout, training=self.training)

        residual = x
        x = F.relu(self.fc1(self.layer_norms[1](x)))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        return residual + F.dropout(x, p=self.res_dropout, training=self.training)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def LayerNorm(embedding_dim):
    return nn.LayerNorm(embedding_dim)
