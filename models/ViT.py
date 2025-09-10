"""
Adapted from https://github.com/lukemelas/simple-bert
"""

import numpy as np
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch
from timm.layers import use_fused_attn
from pytorch_pretrained_vit import ViT


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None  # for visualization
        self.fused_attn = use_fused_attn()

    # def forward(self, x, c=None, mask=None, e=None):
    #     """
    #     x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
    #     mask : (B(batch_size) x S(seq_len))
    #     * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
    #     """
    #     B, N, C = x.shape
    #     if c is None:
    #         c = x
    #     _, M, _ = c.shape

    #     # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
    #     q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

    #     q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
    #     # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
    #     if self.fused_attn:
    #         if mask is not None:
    #             mask = mask[:, None, None, :].bool()
    #         x = F.scaled_dot_product_attention(
    #             q, k, v,
    #             attn_mask=mask,
    #             dropout_p=self.drop.p if self.training else 0.,
    #         )
    #     else:
    #         scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
    #         if mask is not None:
    #             mask = mask[:, None, None, :].float()
    #             scores -= 10000.0 * (1.0 - mask)

    #         if e is not None:
    #             scores = scores + e
    #         scores = self.drop(F.softmax(scores, dim=-1))
    #         # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
    #         h = scores @ v
    #     # -merge-> (B, S, D)
    #     h = x.transpose(1, 2).contiguous().reshape(B, N, C)
    #     # h = merge_last(h, 2)

    #     # self.scores = scores
    #     return h
    
    def forward(self, x, mask=None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))

        # if mask is not None:
        #     mask = mask[:, None, None, :].float()
        #     scores -= 10000.0 * (1.0 - mask)

        scores = self.drop(F.softmax(scores, dim=-1))

        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)

        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Linear(channels[i - 1], channels[i]))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.LayerNorm(channels[i]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff_merge = MLP([dim * 2, dim * 2, dim])
        self.norm2_merge = nn.LayerNorm(dim * 2, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))

        x = x + h
        # x = torch.cat((x,h),dim=2)
        h = self.drop(self.pwff(self.norm2(x)))

        x = x + h
        return x


class transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, num_layers=12, dim=768, num_heads=12, ff_dim=3072, dropout=0.1):
        super(transformer, self).__init__()
        ff_dim = dim * 4
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)

        return x


if __name__ == "__main__":
    image = torch.randn(2, 3, 512, 512)
    descriptors = torch.randn(2, 256, 768)
    vertices_pred = torch.randn(2, 256, 2)
    model = ViT('B_16_imagenet1k', pretrained=False)
    AttenGNN = transformer(num_layers=6, num_heads=6, dim=768)
    outputs = AttenGNN(descriptors)
