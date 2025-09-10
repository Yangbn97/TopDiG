import os

import matplotlib.pyplot as plt
import sys

from torch.cuda.amp import autocast

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
from datetime import datetime
import torch.distributed as dist
import warnings
from utils.setting_utils import *
from models.loss import *
from utils.visualize import *
from utils.poly_utils import *
from utils.metric_utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')


def Train_TCND(epoch, train_step, args, scaler):
    args.detection_model.train()
    # env.ema.eval()
    # update_ema(env.ema, env.detection_model.module, decay=0)
    totalLosses = AverageMeter()
    Losses_node = AverageMeter()
    Losses_edge = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    countt = 0
    recall = {'recall2': [], 'recall5': [], 'recall10': []}
    precision = {'presicion2': [], 'presicion5': [], 'presicion10': []}

    for i, batch in enumerate(args.train_loader):
        start = time.time()
        train_step += 1
        heatmap = batch['heatmaps']
        img = batch['images']
        edge_map_true = batch['boundary_masks']

        image = img.to(device)
        pointmap = heatmap.to(device)
        edge_map_true = edge_map_true.to(device)

        # with autocast():
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            outputs, feature_map = args.detection_model(image)

            if not isinstance(outputs, list):
                outputs = [outputs]
            # save_feature(batch, feature_map, save_dir=env.configs['TrainRoot'])
            gt = pointmap.unsqueeze(1)
            # gt = torch.cat([pointmap.unsqueeze(1), key_pointmap.unsqueeze(1)], dim=1)
            loss_node = args.detection_loss_function(outputs[-1], gt)
            if args.configs['Model']['detection_model'] == 'SAM':
                loss_edge = HDNet_RCF_edge_criterion(outputs[-2], edge_map_true)
                loss = loss_node + loss_edge
            else:
                loss_edge = torch.zeros(1).to(loss_node.device)
                if args.configs['Model']['deep_supervision']:
                    # loss_node += args.detection_loss_function(outputs[-2], pointmap.unsqueeze(1))
                    for o in outputs[:-1]:
                        loss_edge += cross_entropy_loss_RCF(o, edge_map_true)
                    loss = loss_node + loss_edge
                else:
                    loss = loss_node

        batch_out = {'heatmap_pred': F.sigmoid(outputs[-1]),'outputs': [F.sigmoid(out) for out in outputs]}

        # backward
        args.optimizer_detection.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(args.optimizer_detection)
        scaler.update()
        # args.scheduler_detection.step()

        totalLosses.update(loss.item())
        Losses_node.update(loss_node.item())
        Losses_edge.update(loss_edge.item())

        # measure elapsed time
        batch_time.update(time.time() - start)

        if i == 0:
        # if i + 1:
            print('Visualizing......')
            countt += 1
            visualize_keypoints(batch, batch_out, epoch, vis_dir=args.vis_root, N=args.configs['Model']['NUM_POINTS'],
                                mode='train', env=args)
            if args.configs['Model']['show_outputs']:
                visualize_sideouts(batch, batch_out, epoch=epoch, vis_dir=args.vis_root, mode='train')



        # logger.write_hist_parameters(net=model,n_iter=train_step)

        if i % 10 == 0:
            print('Epoch: {} [{}/{} ({:.2f}%)] | trainLoss:{:.6f}, node loss:{:.6f}, edge loss:{:.6f}'.format(
                epoch, i + 1, len(args.train_loader), 100.0 * (i + 1) / len(args.train_loader), loss.item(),
                loss_node.item(), loss_edge.item()))

    curr_detection_lr = args.optimizer_detection.state_dict()['param_groups'][0]['lr']
    args.logger.write_scalars({
        'TrainLoss': totalLosses.avg,
        'NodeLoss': Losses_node.avg,
        'EdgeLoss': Losses_edge.avg,
        'point_lr': curr_detection_lr,
    }, tag='train', n_iter=epoch)

    print('Train Epoch:{} | Loss: {:.4f}'.format(epoch, totalLosses.avg))

    # for key, value in recall.items():
    #     print(key, ':', '{:.4f}'.format(sum(value) / len(value)))
    # for key, value in precision.items():
    #     print(key, ':', '{:.4f}'.format(sum(value) / len(value)))
    return train_step


def Val_TCND(epoch, val_step, best_loss, args):
    args.detection_model.eval()
    with torch.no_grad():
        totalLosses = AverageMeter()
        Losses_node = AverageMeter()
        Losses_edge = AverageMeter()
        countv = 0
        recall = {'recall2': [], 'recall5': [], 'recall10': []}
        precision = {'presicion2': [], 'presicion5': [], 'presicion10': []}

        for i, batch in enumerate(args.val_loader):
            val_step += 1
            heatmap = batch['heatmaps']
            img = batch['images']
            edge_map_true = batch['boundary_masks']

            image = img.to(device)
            pointmap = heatmap.to(device)
            edge_map_true = edge_map_true.to(device)

            # with autocast():
            with torch.autocast(device_type='cuda', dtype=torch.float16):

                outputs, feature_map = args.detection_model(image)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                gt = pointmap.unsqueeze(1)
                # gt = torch.cat([pointmap.unsqueeze(1), key_pointmap.unsqueeze(1)], dim=1)
                loss_node = args.detection_loss_function(outputs[-1], gt)
                if args.configs['Model']['detection_model'] == 'SAM':
                    loss_edge = HDNet_RCF_edge_criterion(outputs[-2], edge_map_true)
                    loss = loss_node + loss_edge
                else:
                    loss_edge = torch.zeros(1).to(loss_node.device)
                    if args.configs['Model']['deep_supervision']:
                        # loss_node += args.detection_loss_function(outputs[-2], pointmap.unsqueeze(1))
                        for o in outputs[:-1]:
                            loss_edge += cross_entropy_loss_RCF(o, edge_map_true)
                        loss = loss_node + loss_edge
                    else:
                        loss = loss_node

            batch_out = {'heatmap_pred': F.sigmoid(outputs[-1]), 'outputs': [F.sigmoid(out) for out in outputs]}
            # batch_out = {'heatmap_pred': F.sigmoid(outputs)}

            totalLosses.update(loss.item())
            Losses_node.update(loss_node.item())
            Losses_edge.update(loss_edge.item())

            # if epoch % 10 == 0 and epoch != 1:
            if epoch + 1:
                vertices_pred, vertices_score = getPoints(outputs[-1], args.configs['Model']['NUM_POINTS'],
                                                          get_score=True, gap=args.gap)
                points_gt = batch['skel_points'] if args.configs['Experiment']['object_type'] == 'line' else batch[
                    'polys']
                b_recall, b_precision = points_detection_acc(vertices_pred, points_gt)

                recall, precision = update_acc(recall, precision, b_recall, b_precision)

            if i == 0 and (epoch - 1) % 3 == 0:
                print('Visualizing......')
                countv += 1
                visualize_keypoints(batch, batch_out, epoch, vis_dir=args.vis_root,
                                    N=args.configs['Model']['NUM_POINTS'], mode='val', env=args)

                if args.configs['Model']['show_outputs']:
                    visualize_sideouts(batch, batch_out, epoch=epoch, vis_dir=args.vis_root, mode='val')

            # if i % int(len(env.val_loader) // 2) == 0:
            if i % 10 == 0:
                print(
                    'Evaluating [{}/{} ({:.2f}%)]'.format(i + 1, len(args.val_loader),
                                                          100.0 * (i + 1) / len(args.val_loader)))

        args.logger.write_scalars({
            'ValLoss': totalLosses.avg,
            'NodeLoss': Losses_node.avg,
            'EdgeLoss': Losses_edge.avg,
        }, tag='val', n_iter=epoch)

        print('Val Epoch:{} | valLoss: {:.4f}, Node Loss: {:.4f}, Edge Loss: {:.4f}'.format(epoch, totalLosses.avg, Losses_node.avg, Losses_edge.avg))
        # if epoch % 10 == 0 and epoch != 1:
        if epoch + 1:
            for key, value in recall.items():
                print(key, ':', '{:.4f}'.format(sum(value) / len(value)))
        #     for key, value in precision.items():
        #         print(key, ':', '{:.4f}'.format(sum(value) / len(value)))
        acc = np.mean(recall['recall5'])
        args.early_stopping([totalLosses, acc], best_loss, pretrain=True)

        if args.early_stopping.save_model:
            best_loss['total_loss'] = totalLosses.avg
            best_loss['acc'] = acc
            save_pretrain_model(args, best_loss, epoch)
            pass
        if args.early_stopping.early_stop:
            best_loss['early_stop'] = True

        print('EarlyStopping counter: {} out of {}'.format(args.early_stopping.counter,
                                                           args.early_stopping.patience))
        if args.early_stopping.counter % 2 == 0 and args.early_stopping.counter > 0:
            print("lr from {} to {}".format(args.optimizer_detection.state_dict()['param_groups'][0]['lr'],
                                            args.optimizer_detection.state_dict()['param_groups'][0]['lr'] * 0.5))
            for p in args.optimizer_detection.param_groups:
                p['lr'] *= 0.5

        return val_step, best_loss
