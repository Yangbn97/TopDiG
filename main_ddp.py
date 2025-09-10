import os

import matplotlib.pyplot as plt
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import time
import datetime
import warnings
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
from diffusers.optimization import get_scheduler
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import builtins
from solvers.train import *
from solvers.valid import *
from solvers.inference import run_inference
from solvers.evaluate import calculate_metrics
from solvers.pretrain_trainer import Train_TCND, Val_TCND
from data.DataLoader import Dataset_Loader_DDP
from models.TCND import get_TCND
from models.TCViT import TCSwin
from models.FPN import FPN
from models.Graph import *
from models.loss import *
from utils.setting_utils import *
from utils.summary import *
from utils.summary import make_print_to_file
from utils.pytorchtools import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')

# 保存原始的 print 函数
original_print = builtins.print

# 判断当前进程是否为 rank 0
def is_rank_0():
    return dist.get_rank() == 0

# 自定义打印函数
def print_if_rank0(*args, **kwargs):
    if is_rank_0():
        original_print(*args, **kwargs)  # 使用原始的 print 函数

setup_seed(20)

def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument(
        '-c', '--configs',
        type=str,
        # default=r'./configs/config_Inria.json',
        # default=r'./configs/config_CrowdAI.json',
        # default=r'./configs/config_DeepGlobe.json',
        default=r'./configs/config_Massachusetts.json',
        # default=r'./configs/config_GID.json',  
        # default=r'./configs/config_BAQS.json',
        # default=r'./configs/config_WAQS.json',
        help='Name of the configs file, excluding the .json file extension.')

    args = parser.parse_args()
    assert args.configs is not None
    cfg = load_config(args.configs)
    args = set_ddp(args, cfg, key='DDP')
    args.configs = cfg
    args.image_size = args.configs['Model']['input_img_size']
    args.class_num = 1
    args.gap = args.configs['Model']['phi']
    args.save_shp = args.configs['Experiment']['save_shp']
    args.save_seg = args.configs['Experiment']['save_seg']
    args.eval = args.configs['Experiment']['infer_eval']
    return args

def main():
    args = get_parser()
    setup_seed(20)
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))

def main_worker(gpu, args):    
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache()

    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    dist.barrier()

    args.save_root = args.configs['Paths']['SaveRoot']
    args.work_dir, args.vis_root, args.model_dir, args.record_root = build_roots(args.configs)

    args.logger = LogSummary(args.work_dir)

    # 替换全局 print
    builtins.print = print_if_rank0

    make_print_to_file(args)
    
    args.early_stopping = EarlyStopping(5, verbose=True, path=args.model_dir)

    if args.configs['Hyper']['batch_size'] >= args.ngpus_per_node:
        args.configs['Hyper']['batch_size'] = int(args.configs['Hyper']['batch_size'] / args.ngpus_per_node)
    args.configs['Hyper']['num_workers'] = int(
        (args.configs['Hyper']['num_workers'] + args.ngpus_per_node - 1) / args.ngpus_per_node)
    args.workers = args.configs['Hyper']['num_workers']

    args.train_loader, args.val_loader = Dataset_Loader_DDP(args)

    if args.configs['Experiment']['pretrain']:
        if args.configs['Model']['detection_model'] == 'TCND':
            args.detection_model = get_TCND(backbone='resnet50', pretrained=True)
        elif args.configs['Model']['detection_model'] == 'TCSwin':
            args.detection_model = TCSwin(image_size=args.image_size)
            # print('Loading pretained SwinTransformer......')
            # weight_path = args.configs['Paths']['pretrained_Swin_weight_path']
            # args.detection_model = load_swin_b_ckpt(weight_path, args.detection_model)
        else:
            args.detection_model = FPN(n_classes=1)

        args.detection_model = to_gpu(args, args.detection_model, mode='DDP')

        args = load_weight(args)

        args.detection_loss_function = torch.nn.MSELoss()
        args.optimizer_detection = torch.optim.Adam(params=args.detection_model.parameters(), lr=args.configs['Hyper']['lr_detection'])

        scaler = torch.cuda.amp.GradScaler()
        print(args.configs)
        best_loss = {'total_loss': 1000, 'point_loss': 1000, 'match_loss': 1000, 'early_stop': False}
        train_step = 0
        val_step = 0
        epoch_num = args.configs['Experiment']['epoch_num']
        for epoch in range(1, epoch_num):
            start_time = datetime.datetime.now()
            print('Training Epoch:{}'.format(epoch))
            train_step = Train_TCND(epoch, train_step, args, scaler)
            print('Validating...')
            val_step, best_loss = Val_TCND(epoch, val_step, best_loss, args)
            end_time = datetime.datetime.now()
            print("Time Elapsed for epoch => {1}".format(epoch, end_time - start_time))
            if best_loss['early_stop']:
                print('Early Stop!')
                break
    else:
        feature_dim=64
        if args.configs['Model']['detection_model'] == 'TCND':
            args.detection_model = get_TCND(backbone='resnet50', pretrained=True)
        elif args.configs['Model']['detection_model'] == 'TCSwin':
            args.detection_model = TCSwin(image_size=args.image_size)
            # print('Loading pretained SwinTransformer......')
            # weight_path = args.configs['Paths']['pretrained_Swin_weight_path']
            # args.detection_model = load_swin_b_ckpt(weight_path, args.detection_model)
        else:
            args.detection_model = FPN(n_classes=1)
            feature_dim=128

        args.match_model = Graph_Generator(sinkhorn=args.configs['Model']['Sinkhorn'], featuremap_dim=feature_dim, configs=args.configs)

        args.detection_loss_function = torch.nn.MSELoss()
        args.match_loss_function = Weighted_BCELoss()
        args.optimizer_detection = torch.optim.Adam(params=args.detection_model.parameters(), lr=args.configs['Hyper']['lr_detection'])
        if args.configs['Model']['Sinkhorn']:
            args.optimizer_match = torch.optim.Adam(params=args.match_model.parameters(), lr=args.configs['Hyper']['lr_match'])
        else:
            args.optimizer_match = torch.optim.AdamW(params=args.match_model.parameters(), lr=args.configs['Hyper']['lr_match'])

        args = load_weight(args)

        args.detection_model = to_gpu(args, args.detection_model, mode='DDP')
        args.match_model = to_gpu(args, args.match_model, mode='DDP')

        if args.configs['Experiment']['infer']:
            run_inference(args)
            if args.eval:
                print('Calculating metrics......')
                calculate_metrics(args)
                return
            return
        scaler = torch.cuda.amp.GradScaler()
        print(args.configs)
        best_loss = {'total_loss': 1000, 'point_loss': 1000, 'match_loss': 1000, 'boundary IoU': 0, 'mask IoU': 0, 'early_stop': False}
        train_step = 0
        val_step = 0
        if args.configs['Experiment']['evaluate']:
            print('Evaluating......')
            val_step, best_loss = run_val(1, val_step, best_loss, args)
            return
        else:
            epoch_num = args.configs['Experiment']['epoch_num']
        for epoch in range(1, epoch_num):
            start_time = datetime.datetime.now()
            if not args.configs['Experiment']['evaluate']:
                print('Training Epoch:{}'.format(epoch))
                train_step = run_train(epoch, train_step, args, scaler)
            print('Validating...')
            val_step, best_loss = run_val(epoch, val_step, best_loss, args)
            end_time = datetime.datetime.now()
            print("Time Elapsed for epoch => {1}".format(epoch, end_time - start_time))
            if best_loss['early_stop']:
                print('Early Stop!')
                break



if __name__ == '__main__':
    main()
