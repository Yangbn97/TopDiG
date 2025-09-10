import os
import matplotlib.pyplot as plt
import sys
os.environ['NCCL_P2P_LEVEL'] = 'NVL'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
import time
import datetime
import warnings
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
# from diffusers.optimization import get_scheduler
import torch.nn.functional as F
from solvers.train import *
from solvers.valid import *
from solvers.inference import run_inference
from solvers.evaluate import calculate_metrics
from solvers.pretrain_trainer import Train_TCND, Val_TCND
from data.DataLoader import Dataset_Loader
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

setup_seed(20)
parser = argparse.ArgumentParser(description='Params for label making')
parser.add_argument(
    '-c', '--configs_path',
    # type=str,
    # default=r'./configs/config_GID.json',
    # default=r'./configs/config_Massachusetts.json',
    # default=r'./configs/config_DeepGlobe.json',
    default=r'./configs/config_CrowdAI.json',
    # default=r'./configs/config_Inria.json',
    help='Name of the configs file, excluding the .json file extension.')


args = parser.parse_args()
assert args.configs_path is not None, "Argument --configs must be specified. Run 'python main.py --help' for help on arguments."
configs = load_config(args.configs_path)
args.configs = configs
args.image_size = args.configs['Model']['input_img_size']
args.gap = args.configs['Model']['phi']
args.class_num = 1
args.save_root = args.configs['Paths']['SaveRoot']
args.configs['DDP']['flag'] = 0
args.work_dir, args.vis_root, args.model_dir, args.record_root = build_roots(configs)

args.logger = LogSummary(args.work_dir)

make_print_to_file(args)
args.early_stopping = EarlyStopping(10, verbose=True, path=args.model_dir)


args.save_shp = args.configs['Experiment']['save_shp']
args.save_seg = args.configs['Experiment']['save_seg']
args.eval = args.configs['Experiment']['infer_eval']


if not args.configs['Experiment']['infer']:
    args.train_loader, args.val_loader = Dataset_Loader(configs)
else:
    pass

def main(args):
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

        args = load_weight(args)
        
        args.detection_model = to_gpu(args, args.detection_model, mode='DP')


        args.detection_loss_function = torch.nn.MSELoss()
        args.optimizer_detection = torch.optim.Adam(params=args.detection_model.parameters(), lr=args.configs['Hyper']['lr_detection'])

        scaler = torch.cuda.amp.GradScaler()
        print(args.configs)
        best_loss = {'total_loss': 1000, 'point_loss': 1000, 'acc': 0, 'early_stop': False}
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
        weight = args.configs['Model']['machL_weight'] if args.configs['Experiment']['dataset_name'] in  ['DeepGlobe','Massachusetts'] else None
        args.match_loss_function = Weighted_BCELoss(weight)
        args.optimizer_detection = torch.optim.Adam(params=args.detection_model.parameters(), lr=args.configs['Hyper']['lr_detection'])
        if args.configs['Model']['Sinkhorn']:
            args.optimizer_match = torch.optim.Adam(params=args.match_model.parameters(), lr=args.configs['Hyper']['lr_match'])
        else:
            args.optimizer_match = torch.optim.AdamW(params=args.match_model.parameters(), lr=args.configs['Hyper']['lr_match'])

        args = load_weight(args)
        args.detection_model = to_gpu(args, args.detection_model, mode='DP')
        args.match_model = to_gpu(args, args.match_model, mode='DP')
        
        if configs['Experiment']['infer']:
            run_inference(args)
            if args.eval:
                print('Calculating metrics......')
                calculate_metrics(args)
                return
            return
        scaler = torch.cuda.amp.GradScaler()
        print(configs)
        best_loss = {'total_loss': 1000, 'point_loss': 1000, 'match_loss': 1000, 'boundary IoU': 0, 'mask IoU': 0, 'early_stop': False}
        train_step = 0
        val_step = 0
        if args.configs['Experiment']['evaluate']:
            print('Evaluating......')
            run_eval(args)
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
    main(args)
