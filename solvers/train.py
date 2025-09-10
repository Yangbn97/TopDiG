import os

import matplotlib.pyplot as plt
import sys

from torch.cuda.amp import autocast

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
from datetime import datetime
import warnings
import argparse
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
import torch.nn.functional as F
from utils.visualize import *
from utils.poly_utils import *
from utils.metric_utils import *
from utils.DGS_utils import create_adjacency_matrix_from_skeleton
from utils.pytorchtools import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')


def run_train(epoch, train_step, args, scaler):
    args.detection_model.train()
    args.match_model.train()

    totalLosses = AverageMeter()
    losses_point = AverageMeter()
    losses_match = AverageMeter()
    mask_IoUs = AverageMeter()
    boundary_IoUs = AverageMeter()
    stats = {'Mask': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []},
             'Boundary': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}}

    countt = 0

    for i, batch in enumerate(args.train_loader):
        start = time.time()
        train_step += 1
        heatmap = batch['heatmaps']
        img = batch['images']


        mask_true = batch['masks']
        boundary_mask_true = batch['boundary_masks']
        image = img.to(device)
        pointmap = heatmap.to(device)
        mask_true = mask_true.to(device)
        boundary_mask_true = boundary_mask_true.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):

            heatmap_pred, feature_map = args.detection_model(image)
            if isinstance(heatmap_pred, list):
                heatmap_pred = heatmap_pred[-1]
            loss_points = args.detection_loss_function(heatmap_pred, pointmap)

            vertices_pred, vertices_score = getPoints(heatmap_pred, args.configs['Model']['NUM_POINTS'], get_score=True, gap=args.gap)


            S, PM_pred = args.match_model(image, feature_map, vertices_pred)

            batch_out = {'heatmap_pred': heatmap_pred, 'points_pred': vertices_pred,
                         'vertices_score': vertices_score,'PM_pred': PM_pred}


            if args.configs['Experiment']['object_type'] == 'line':
                # PM_gt = getAdjMatrix_road(vertices_pred, vertices_score, batch['skel_points'], batch['junctions'],tol=args.configs['Model']['delta'])
                PM_gt = create_adjacency_matrix_from_skeleton(vertices_pred, vertices_score, mask_true, dist_threshold=args.configs['Model']['delta'])
                PM_true = torch.from_numpy(PM_gt).unsqueeze(1)
                PM_true = PM_true.to(device)
                loss_match = args.match_loss_function(S.unsqueeze(1), PM_true)
            else:
                polys = batch['polys']
                PM_gt = getAdjMatrix_poly(vertices_pred, vertices_score, polys, score=0.6,tol=args.configs['Model']['delta'])
                PM_true = torch.from_numpy(PM_gt).unsqueeze(1)
                PM_true = PM_true.to(device)
                loss_match = -torch.mean(torch.masked_select(PM_pred.unsqueeze(1), PM_true == 1))

            batch['PM_label'] = PM_true

            loss = loss_match + loss_points

        # backward
        args.optimizer_detection.zero_grad()
        args.optimizer_match.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(args.optimizer_detection)
        scaler.step(args.optimizer_match)
        scaler.update()


        if args.configs['Experiment']['object_type'] == 'line':
            mask_pred, boundary_mask_pred = PM2road(batch, batch_out, epoch=epoch, vis=False,
                                                    vis_dir=args.vis_root,
                                                    mode='PM2Road', threshold=args.configs['Model']['threshold'],score=0.5)

            # stats_boundary = performMetrics(boundary_mask_pred, boundary_mask_true.squeeze(1))
            # stats_mask = stats_boundary
            # boundary_mask_iou = stats_boundary['IoU']
            # mask_iou = boundary_mask_iou
            # stats = update_stats(stats, stats_mask, key='Mask')
            # stats = update_stats(stats, stats_boundary, key='Boundary')
        else:
            mask_pred, boundary_mask_pred = PM2poly(batch, batch_out, epoch=epoch, vis=False,
                                                    vis_dir=args.vis_root, mode='PM2Poly')

            # stats_mask = performMetrics(mask_pred, mask_true.squeeze(1))
            # mask_iou = stats_mask['IoU']

            # stats_boundary = performMetrics(boundary_mask_pred, boundary_mask_true.squeeze(1))
            # boundary_mask_iou = stats_boundary['IoU']
            # stats = update_stats(stats, stats_mask, key='Mask')
            # stats = update_stats(stats, stats_boundary, key='Boundary')

        # if i == 0:
        #     visualize_Adjacency_Matrix(S, PM_pred, PM_true, args, epoch, batch)

        totalLosses.update(loss.item())
        losses_point.update(loss_points.item())
        losses_match.update(loss_match.item())
        # mask_IoUs.update(mask_iou)
        # boundary_IoUs.update(boundary_mask_iou)


        if i == 0 and epoch == 1:
            print('Visualizing......')
            if args.configs['Experiment']['object_type'] == 'line':
                visualize_output_road(batch, batch_out, epoch, vis_dir=args.vis_root,
                                             mode='train',
                                             threshold=args.configs['Model']['threshold'],score=0.5)
            else:
                visualize_output_poly(batch, batch_out, epoch, vis_dir=args.vis_root,mode='train')


        if i % 10 == 0:
            print(
                'Epoch: {} [{}/{} ({:.2f}%)] | trainLoss:{:.6f}, Point Loss: {:.6f}, Match Loss: {:.6f}'.format(
                    epoch, i + 1, len(args.train_loader), 100.0 * (i + 1) / len(args.train_loader), loss.item(),
                    loss_points.item(),
                    loss_match.item()))


    curr_detection_lr = args.optimizer_detection.state_dict()['param_groups'][0]['lr']
    curr_match_lr = args.optimizer_match.state_dict()['param_groups'][0]['lr']
    # args.logger.write_scalars({
    #     'loss_train': totalLosses.avg,
    #     'loss_point': losses_point.avg,
    #     'loss_match': losses_match.avg,
    #     'mask_iou': mask_IoUs.avg,
    #     'boundary_iou': boundary_IoUs.avg,
    #     'point_lr': curr_detection_lr,
    #     'match_lr': curr_match_lr
    # }, tag='train', n_iter=epoch)

    print(
        'Train Epoch:{} | Loss: {:.4f}, Point Loss: {:.4f},Match Loss: {:.4f},mask_IoU: {:.6f},boundary_IoU: {:.6f}'.format(
            epoch,
            totalLosses.avg,
            losses_point.avg,
            losses_match.avg,
            mask_IoUs.avg,
            boundary_IoUs.avg))

    # summary_stats(stats)

    del loss_match, loss_points, loss
    return train_step