
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchinfo import summary
from thop import profile,clever_format
from models.base import BaseNet
import torch.nn.functional as F

__all__ = ['DFF', 'get_dff']

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DetectionBranch(nn.Module):
    def __init__(self, in_dim=64, n_classes=1):
        super(DetectionBranch,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1,stride=1,padding=0,bias=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self,in_dim,out_dim,pad=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=pad)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class TCND(BaseNet):
    r"""Dynamic Feature Fusion for Semantic Edge Detection

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Yuan Hu, Yunpeng Chen, Xiang Li, Jiashi Feng. "Dynamic Feature Fusion
        for Semantic Edge Detection" *IJCAI*, 2019

    """

    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, **kwargs):
        super(TCND, self).__init__(nclass, backbone, norm_layer=norm_layer, **kwargs)
        self.nclass = nclass

        self.ada_learner = LocationAdaptiveLearner(nclass, nclass * 4, nclass * 4, norm_layer=norm_layer)

        # V1
        self.side1 = nn.Sequential(nn.Conv2d(64, 1, 1),
                                   norm_layer(1))
        self.side2 = nn.Sequential(nn.Conv2d(256, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side3 = nn.Sequential(nn.Conv2d(512, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        self.side5 = nn.Sequential(nn.Conv2d(2048, nclass, 1, bias=True),
                                   norm_layer(nclass),
                                   nn.ConvTranspose2d(nclass, nclass, 16, stride=8, padding=4, bias=False))

        self.side5_w = nn.Sequential(nn.Conv2d(2048, nclass * 4, 1, bias=True),
                                     norm_layer(nclass * 4),
                                     nn.ConvTranspose2d(nclass * 4, nclass * 4, 16, stride=8, padding=4, bias=False))

        self.featureconv1 = ConvBlock(self.nclass * 4, 64)
        self.featureconv2 = ConvBlock(64, 64)
        self.seg_head = DetectionBranch(in_dim=64, n_classes=nclass)

    @autocast()
    def forward(self, x, return_sides=True):
        _, _, h, w = x.size()
        c1, c2, c3, _, c5 = self.base_forward(x)

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


        ada_weights = self.ada_learner(side5_w)  # (N, nclass, 4, H, W)

        fuse_feature = self.featureconv1(fuse)
        fuse_feature = self.featureconv2(fuse_feature)


        fuse = self.seg_head(fuse_feature)


        final = fuse.view(fuse.size(0), self.nclass, -1, fuse.size(2), fuse.size(3))  # (N, nclass, 4, H, W)
        final = torch.mul(final, ada_weights)  # (N, nclass, 4, H, W)
        final = torch.sum(final, 2)  # (N, nclass, H, W)

        if return_sides:
            return [side1,side2,side3,side5,final],fuse_feature
        else:
            return fuse, fuse_feature

class LocationAdaptiveLearner(nn.Module):
    """docstring for LocationAdaptiveLearner"""

    def __init__(self, nclass, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(LocationAdaptiveLearner, self).__init__()
        self.nclass = nclass

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels))

    def forward(self, x):
        # x:side5_w (N, n_class*4, H, W)
        x = self.conv1(x)  # (N, n_class*4, H, W)
        x = self.conv2(x)  # (N, n_class*4, H, W)
        x = self.conv3(x)  # (N, n_class*4, H, W)
        x = x.view(x.size(0), self.nclass, -1, x.size(2), x.size(3))  # (N, n_class, 4, H, W)
        return x


def get_TCND(backbone='resnet50', pretrained=False,
            root='./pretrain_models', **kwargs):

    # infer number of classes
    model = TCND(nclass=1,pretrained=pretrained, backbone=backbone, root=root, **kwargs)

    return model

if __name__ == "__main__":
    input = torch.randn(1,3,512,512)
    model = get_TCND(backbone='resnet50', pretrained=False)
    # summary(model, input_size=(1, 3, 300, 300))
    flops, params = profile(model, inputs=(input,))
    macs, params_ = clever_format([flops, params], "%.3f")
    print('MACs:', macs)
    print('Paras:', params_)
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    # outputs = model(input)
    # print(outputs[-1].shape)