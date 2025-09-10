import os

import matplotlib.pyplot as plt
import sys

from torch.cuda.amp import autocast
import torch.distributed as dist
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
from utils.DGS_utils import create_adjacency_matrix_from_skeleton
from utils.setting_utils import *
from utils.metric_utils import *
from utils.pytorchtools import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')

def run_val(epoch, val_step, best_loss, args):
    args.detection_model.eval()
    args.match_model.eval()
    with torch.no_grad():

        totalLosses = AverageMeter()
        losses_point = AverageMeter()
        losses_match = AverageMeter()
        mask_IoUs = AverageMeter()
        boundary_IoUs = AverageMeter()
        stats = {'Mask': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []},
                 'Boundary': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []},
                 'Points': {'recall2': [], 'recall5': [], 'recall10': [],
                            'precision2': [], 'precision5': [], 'precision10': []}}
        stats_5classes = {'Mask': {'1': [], '2': [], '3': [], '4': [], '5': []},
                 'Boundary': {'1': [], '2': [], '3': [], '4': [], '5': []}}
        recall = {'recall2': [], 'recall5': [], 'recall10': []}
        precision = {'precision2': [], 'precision5': [], 'precision10': []}

        # if args.configs['Experiment']['object_type'] != 'road':
        #     stats['Boundary']['MTA'] = []

        vis_flag = False



        for i, batch in enumerate(args.val_loader):
            start = time.time()
            val_step += 1
            heatmap = batch['heatmaps']
            img = batch['images']
            mask_true = batch['masks']
            boundary_mask_true = batch['boundary_masks']
            names = batch['names']

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
                             'vertices_score': vertices_score, 'PM_pred': PM_pred}


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

            if args.configs['Experiment']['object_type'] == 'line':
                mask_pred, boundary_mask_pred = PM2road(batch, batch_out, epoch=epoch, vis=vis_flag,
                                                        vis_dir=args.vis_root,
                                                        mode='PM2Road', threshold=args.configs['Model']['threshold'], score=0.6)

                # output_road(batch, batch_out, vis=True, vis_dir=args.vis_root, threshold=args.configs['Model']['threshold'],
                #             score=0.6, line_thickness=7)

                stats_boundary = performMetrics(boundary_mask_pred, boundary_mask_true.squeeze(1))
                stats_mask = stats_boundary
                boundary_mask_iou = stats_boundary['IoU']
                mask_iou = boundary_mask_iou
                stats = update_stats(stats, stats_mask, key='Mask')
                stats = update_stats(stats, stats_boundary, key='Boundary')
            else:
                mask_pred, boundary_mask_pred, polys_pred = PM2poly(batch, batch_out, epoch=epoch, vis=vis_flag,
                                                        vis_dir=args.vis_root,mode='PM2Poly', retrun_polys=True)

                # output_poly(batch, batch_out, vis=True,vis_dir=args.vis_root, line_thickness=7)

                stats_mask = performMetrics(mask_pred, mask_true.squeeze(1))
                mask_iou = stats_mask['IoU']

                stats_boundary = performMetrics(boundary_mask_pred, boundary_mask_true.squeeze(1))

                boundary_mask_iou = stats_boundary['IoU']
                stats = update_stats(stats, stats_mask, key='Mask')
                stats = update_stats(stats, stats_boundary, key='Boundary')

                # label_5classes_path = os.path.join(mulclasses_dir, '{}.tif'.format(names[0]))
                # label_5classes = cv2.imread(label_5classes_path, 0)
                # cls_list = np.unique(label_5classes)
                # cls_list = cls_list[cls_list != 0]
                # cls_num = len(cls_list)
                # stats_5classes['Mask'][str(cls_num)].append(mask_iou)
                # stats_5classes['Boundary'][str(cls_num)].append(boundary_mask_iou)


            totalLosses.update(loss.item())
            losses_point.update(loss_points.item())
            losses_match.update(loss_match.item())
            mask_IoUs.update(mask_iou)
            boundary_IoUs.update(boundary_mask_iou)

            points_gt = batch['skel_points'] if args.configs['Experiment']['object_type'] == 'line' else batch['polys']
            b_recall, b_precision = points_detection_acc(vertices_pred, points_gt)

            recall, precision = update_acc(recall, precision, b_recall, b_precision)

            points_stats = {**b_recall, **b_precision}
            stats = update_stats(stats, points_stats, key='Points')


            if i == 0 and not args.configs['Experiment']['evaluate']:
            # if i + 1:
                visualize_Adjacency_Matrix(S, PM_pred, PM_true, args, epoch, batch)

            if not args.configs['Experiment']['evaluate'] and i == 0:
            # if args.configs['Experiment']['evaluate']:
                # title = f'mask_IoU={round(mask_iou,3)}% bdy_IoU={round(boundary_mask_iou,3)}% cls_num={cls_num}'
                print('Visualizing [{}/{} ({:.2f}%)]'.format(i + 1, len(args.val_loader),
                                                             100.0 * (i + 1) / len(args.val_loader)))
                mode = 'val'
                if args.configs['Experiment']['evaluate']:
                    mode = 'eval'
                if args.configs['Experiment']['object_type'] == 'line':
                    visualize_output_road(batch, batch_out, epoch, vis_dir=args.vis_root,
                                          mode=mode,
                                          threshold=args.configs['Model']['threshold'], score=0.6)
                else:
                    visualize_output_poly(batch, batch_out, epoch, vis_dir=args.vis_root, mode=mode)

            if i % int(len(args.val_loader) // 5) == 0:
                print(
                    'Evaluating [{}/{} ({:.2f}%)]'.format(i + 1, len(args.val_loader),
                                                          100.0 * (i + 1) / len(args.val_loader)))

        if not args.configs['Experiment']['evaluate']:
            args.logger.write_scalars({
                'loss_valid': totalLosses.avg,
                'loss_point': losses_point.avg,
                'loss_match': losses_match.avg,
                'mask_iou': mask_IoUs.avg,
                'boundary_iou': boundary_IoUs.avg,
            }, tag='val', n_iter=epoch)

        print('Valid Epoch:{} | Loss: {:.4f}, Mask IoU: {:.4f}, boundary IoU: {:.4f}'.format(epoch, totalLosses.avg, mask_IoUs.avg, boundary_IoUs.avg))

        summary_stats(stats)
        # summary_stats(stats_5classes)

        if not args.configs['Experiment']['evaluate']:
            current_loss = [totalLosses, losses_point, losses_match, boundary_IoUs, mask_IoUs]#
            # best_loss = save_model(args, best_loss, [totalLosses, losses_point, losses_match, boundary_IoUs, mask_IoUs],
            #                        epoch)

            args.early_stopping(current_loss, best_loss, pretrain=False)
            if args.early_stopping.save_model and not args.configs['Experiment']['evaluate']:
                if args.configs['DDP']['flag']:
                    if dist.get_rank() == 0:
                        best_loss = save_model(args, best_loss, [totalLosses, losses_point, losses_match, boundary_IoUs, mask_IoUs], epoch)
                else:
                    best_loss = save_model(args, best_loss, [totalLosses, losses_point, losses_match, boundary_IoUs, mask_IoUs], epoch)
            if args.early_stopping.early_stop:
                best_loss['early_stop'] = True

            # if args.configs['Experiment']['object_type'] == 'line':
            print('EarlyStopping counter: {} out of {}'.format(args.early_stopping.counter,
                                                                args.early_stopping.patience))
            if args.early_stopping.counter % 4 == 0 and args.early_stopping.counter > 0:
                print("lr from {} to {}".format(args.optimizer_match.state_dict()['param_groups'][0]['lr'],
                                                args.optimizer_match.state_dict()['param_groups'][0]['lr'] * 0.5))
                for p in args.optimizer_match.param_groups:
                    p['lr'] *= 0.5

        return val_step, best_loss
    
def run_eval(args):
    args.detection_model.eval()
    args.match_model.eval()
    with torch.no_grad():

        totalLosses = AverageMeter()
        losses_point = AverageMeter()
        losses_match = AverageMeter()
        mask_IoUs = AverageMeter()
        boundary_IoUs = AverageMeter()
        stats = {'Mask': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []},
                 'Boundary': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []},
                 'Points': {'recall2': [], 'recall5': [], 'recall10': [],
                            'precision2': [], 'precision5': [], 'precision10': []}}
        recall = {'recall2': [], 'recall5': [], 'recall10': []}
        precision = {'precision2': [], 'precision5': [], 'precision10': []}

        vis_flag = False



        for i, batch in enumerate(args.val_loader):
            start = time.time()
            heatmap = batch['heatmaps']
            img = batch['images']
            mask_true = batch['masks']
            boundary_mask_true = batch['boundary_masks']
            names = batch['names']

            image = img.to(device)
            pointmap = heatmap.to(device)
            mask_true = mask_true.to(device)
            boundary_mask_true = boundary_mask_true.to(device)

            with autocast():

                heatmap_pred, feature_map = args.detection_model(image)
                if isinstance(heatmap_pred, list):
                    heatmap_pred = heatmap_pred[-1]

                vertices_pred, vertices_score = getPoints(heatmap_pred, args.configs['Model']['NUM_POINTS'], get_score=True, gap=args.gap)

                S, PM_pred = args.match_model(image, feature_map, vertices_pred)

                batch_out = {'heatmap_pred': heatmap_pred, 'points_pred': vertices_pred,
                             'vertices_score': vertices_score, 'PM_pred': PM_pred}


                if args.configs['Experiment']['object_type'] == 'line':
                    # PM_gt = getAdjMatrix_road(vertices_pred, vertices_score, batch['skel_points'], batch['junctions'],tol=args.configs['Model']['delta'])
                    PM_gt = create_adjacency_matrix_from_skeleton(vertices_pred, vertices_score, mask_true, dist_threshold=args.configs['Model']['delta'])
                    PM_true = torch.from_numpy(PM_gt).unsqueeze(1)
                    PM_true = PM_true.to(device)
                else:
                    polys = batch['polys']
                    PM_gt = getAdjMatrix_poly(vertices_pred, vertices_score, polys, score=0.6,tol=args.configs['Model']['delta'])
                    PM_true = torch.from_numpy(PM_gt).unsqueeze(1)
                    PM_true = PM_true.to(device)

                batch['PM_label'] = PM_true

            if args.configs['Experiment']['object_type'] == 'line':
                mask_pred, boundary_mask_pred = PM2road(batch, batch_out, epoch=1, vis=vis_flag,
                                                        vis_dir=args.vis_root,
                                                        mode='PM2Road', threshold=args.configs['Model']['threshold'], score=0.6)

                # output_road(batch, batch_out, vis=True, vis_dir=args.vis_root, threshold=args.configs['Model']['threshold'],
                #             score=0.6, line_thickness=7)

                stats_boundary = performMetrics(boundary_mask_pred, boundary_mask_true.squeeze(1))
                stats_mask = stats_boundary
                boundary_mask_iou = stats_boundary['IoU']
                mask_iou = boundary_mask_iou
                stats = update_stats(stats, stats_mask, key='Mask')
                stats = update_stats(stats, stats_boundary, key='Boundary')
            else:
                mask_pred, boundary_mask_pred, polys_pred = PM2poly(batch, batch_out, epoch=1, vis=vis_flag,
                                                        vis_dir=args.vis_root,mode='PM2Poly', retrun_polys=True)

                batch_out['mask_pred'] = mask_pred
                batch_out['boundary_mask_pred'] = boundary_mask_pred
                output_poly(batch, batch_out, vis=True,vis_dir=args.vis_root)

                stats_mask = performMetrics(mask_pred, mask_true.squeeze(1))
                mask_iou = stats_mask['IoU']

                stats_boundary = performMetrics(boundary_mask_pred, boundary_mask_true.squeeze(1))

                boundary_mask_iou = stats_boundary['IoU']
                stats = update_stats(stats, stats_mask, key='Mask')
                stats = update_stats(stats, stats_boundary, key='Boundary')

            mask_IoUs.update(mask_iou)
            boundary_IoUs.update(boundary_mask_iou)

            points_gt = batch['skel_points'] if args.configs['Experiment']['object_type'] == 'line' else batch['polys']
            b_recall, b_precision = points_detection_acc(vertices_pred, points_gt)

            recall, precision = update_acc(recall, precision, b_recall, b_precision)

            points_stats = {**b_recall, **b_precision}
            stats = update_stats(stats, points_stats, key='Points')

            if i % int(len(args.val_loader) // 5) == 0:
                print(
                    'Evaluating [{}/{} ({:.2f}%)]'.format(i + 1, len(args.val_loader),
                                                          100.0 * (i + 1) / len(args.val_loader)))
                
            print('Visualizing [{}/{} ({:.2f}%)]'.format(i + 1, len(args.val_loader),
                                                            100.0 * (i + 1) / len(args.val_loader)))
            mode = 'eval'
            if args.configs['Experiment']['object_type'] == 'line':
                visualize_road_eval(batch, batch_out, vis_dir=args.vis_root,
                                        mode=mode,
                                        threshold=args.configs['Model']['threshold'], score=0.6)
            else:
                visualize_poly_eval(batch, batch_out, vis_dir=args.vis_root, mode=mode)


        summary_stats(stats)
