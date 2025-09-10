import torch
import torch.nn as nn
import torch.nn.functional as F

class Weighted_BCELoss(nn.Module):
    def __init__(self, weight=None):
        super(Weighted_BCELoss, self).__init__()
        self.weight = weight

    def forward(self, prediction, target):
        pos_weight = 0.7 if self.weight is None else self.weight[0]
        neg_weight = 0.3 if self.weight is None else self.weight[1]
        # print(pos_weight)
        # bce_loss = nn.BCELoss(size_average=True)
        # prediction = F.sigmoid(prediction)
        weight = torch.zeros_like(target)
        weight = torch.fill_(weight, neg_weight)
        weight[target > 0] = pos_weight
        loss = nn.BCEWithLogitsLoss(weight=weight, size_average=True)(prediction, target.float())
        return loss

def cross_entropy_loss_RCF(prediction, labelf):
    label = labelf.long()
    mask = labelf.float()
    num_positive = torch.sum((label==1).float()).float()
    num_negative = torch.sum((label==0).float()).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy_with_logits(
            prediction.float(),labelf.float(), weight=mask, reduction='mean')
    return torch.mean(cost)

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()

def HDNet_RCF_edge_criterion(inputs, target):
    loss1 = cross_entropy_loss_RCF(inputs, target)
    loss2 = dice_loss_func(F.sigmoid(inputs.squeeze()), target.squeeze().float())
    return loss1 + loss2
