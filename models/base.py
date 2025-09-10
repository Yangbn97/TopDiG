import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import upsample
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter
from thop import profile
from typing import Type
from models import resnet

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, norm_layer=None,
                 crop_size=472, mean=[.485, .456, .406],
                 std=[.229, .224, .225], pretrained=False,root='./pretrain_models'):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.mean = mean
        self.std = std
        self.crop_size = crop_size
        self.pretrained = pretrained
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=self.pretrained,
                                              norm_layer=norm_layer, root=root)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=self.pretrained,
                                               norm_layer=norm_layer, root=root)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=self.pretrained,
                                               norm_layer=norm_layer, root=root)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs


    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        c1 = self.pretrained.relu(x)

        x = self.pretrained.maxpool(c1)
        c2 = self.pretrained.layer1(x)
        c3 = self.pretrained.layer2(c2)
        c4 = self.pretrained.layer3(c3)
        c5 = self.pretrained.layer4(c4)
        return c1, c2, c3, c4, c5
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    
if __name__ == "__main__":
    input = torch.randn(1,3,300,300)
    model = BaseNet(nclass=1,backbone='resnet50', norm_layer=nn.BatchNorm2d)
    # summary(model, input_size=(1, 3, 300, 300))
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')