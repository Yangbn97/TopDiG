"""
Adapted from https://github.com/lukemelas/simple-bert
"""

import numpy as np
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch
from einops import rearrange, repeat
# from pytorch_pretrained_vit import ViT
from models.TCND import ConvBlock, DetectionBranch, LocationAdaptiveLearner
from models.swin_transformer import swin_b, Swin_B_Weights

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

    def forward(self, x, mask):
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

    def forward(self, x, mask=None, return_reprs=False):
        outputs = []
        for block in self.blocks:
            x = block(x, mask)
            outputs.append(x)
        if return_reprs:
            return outputs
        else:
            return x

def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class TCSwin(nn.Module):
    def __init__(self, n_class=1, image_size=320, patches=8,
                 num_layers=12, in_channels=3, dim=768,
                 num_heads=12, ff_dim=3072, dropout=0.1,
                 positional_embedding='1d', return_reprs=True):
        super(TCSwin, self).__init__()
        self.image_size = image_size
        self.return_reprs = return_reprs
        self.nclass = n_class

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        self.patch_emd = (gh,gw)
        seq_len = gh * gw
        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()

        # Transformer
        # self.transformer = transformer(num_layers=num_layers, dim=dim, num_heads=num_heads,
        #                                ff_dim=ff_dim, dropout=dropout)
        self.transformer = swin_b(weights=Swin_B_Weights)

        self.ada_learner = LocationAdaptiveLearner(n_class, n_class * 4, n_class * 4, norm_layer=nn.BatchNorm2d)

        self.side1 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1))

        self.side2 = nn.Sequential(nn.ConvTranspose2d(256, 256, 8, stride=4, padding=2),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(256, 1, 4, stride=2, padding=1))

        self.side3 = nn.Sequential(nn.ConvTranspose2d(512, 512, 8, stride=4, padding=2),
                                   nn.BatchNorm2d(512),
                                   nn.ConvTranspose2d(512, 1, 8, stride=4, padding=2))
        self.side5 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 8, stride=4, padding=2),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(512, 256, 8, stride=4, padding=2),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(256, n_class, 4, stride=2, padding=1))

        self.side5_w = nn.Sequential(nn.ConvTranspose2d(1024, 512, 8, stride=4, padding=2),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(512, 256, 8, stride=4, padding=2),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(256, n_class*4, 4, stride=2, padding=1))

        self.featureconv1 = ConvBlock(self.nclass * 4, 64)
        self.featureconv2 = ConvBlock(64, 64)
        n_classes = 1 if self.nclass == 1 else self.nclass * 4
        self.seg_head = DetectionBranch(in_dim=64, n_classes=n_classes)

        # Initialize weights
        self.init_weights()
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)
        # nn.init.constant_(self.fc.weight, 0)
        # nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding,
                        std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        # nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        # x = self.patch_embedding(x)  # b,d,gh,gw
        # x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        # if hasattr(self, 'positional_embedding'):
        #     x = self.positional_embedding(x)  # b,gh*gw,d

        vis = self.transformer(x)  # b,gh*gw,d

        c1, c2, c3, c5 = vis[0], vis[1], vis[2], vis[-1]
        side1 = self.side1(c1)  # (N, 1, H, W)
        side2 = self.side2(c2)  # (N, 1, H, W)
        side2 = F.interpolate(side2, (h, w), mode='bilinear')
        side3 = self.side3(c3)  # (N, 1, H, W)
        side3 = F.interpolate(side3, (h, w), mode='bilinear')
        side5 = self.side5(c5)  # (N, nclass, H, W)
        side5 = F.interpolate(side5, (h, w), mode='bilinear')
        side5_w = self.side5_w(c5)  # (N, nclass*4, H, W)
        side5_w = F.interpolate(side5_w, (h, w), mode='bilinear')

        slice5 = side5[:, 0:1, :, :]  # (N, 1, H, W)
        fuse = torch.cat((slice5, side1, side2, side3), 1)
        for i in range(side5.size(1) - 1):
            slice5 = side5[:, i + 1:i + 2, :, :]  # (N, 1, H, W)
            fuse = torch.cat((fuse, slice5, side1, side2, side3), dim=1)  # (N, nclass*4, H, W)

        ada_weights = self.ada_learner(side5_w)  # (N, nclass, 4, H, W)

        fuse_feature = self.featureconv1(fuse)
        fuse_feature = self.featureconv2(fuse_feature)

        fuse = self.seg_head(fuse_feature)  # (N, nclass*4, H, W)

        fuse = fuse.view(fuse.size(0), self.nclass, -1, fuse.size(2), fuse.size(3))  # (N, nclass, 4, H, W)
        fuse = torch.mul(fuse, ada_weights)  # (N, nclass, 4, H, W)
        fuse = torch.sum(fuse, 2)  # (N, nclass, H, W)

        return [side1, side2, side3, side5, fuse], fuse_feature



if __name__ == "__main__":
    image = torch.randn(2, 3, 300, 300)
    descriptors = torch.randn(2, 256, 768)
    vertices_pred = torch.randn(2, 256, 2)
    model = TCSwin(n_class=1, patches=8, num_layers=12, num_heads=6, dim=768)
    outs, f = model(image)
    # AttenGNN = transformer(num_layers=6, num_heads=6, dim=768)
    # outputs = AttenGNN(descriptors)
