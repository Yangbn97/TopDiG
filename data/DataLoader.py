import os

import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.utils.data as data
from functools import partial
from data.dataset_building import *
from data.dataset_Massachusetts import *
from data.dataset_water import *
from data.dataset_map_challenge import *

def Dataset_Loader(configs):
    train_ROOT = configs['Paths']['TrainRoot']
    val_ROOT = configs['Paths']['ValRoot']
    NUM_POINTS = configs['Model']['NUM_POINTS']
    dilate_pixels = configs['Model']['dilate_pixels']
    train_loader = []
    val_loader = []
    if configs['Experiment']['dataset_name'] == 'Inria':
        if not configs['Experiment']['evaluate']:
            trainset = Dataset_Inria(train_ROOT, mode='train', N=NUM_POINTS, dilate=dilate_pixels)
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=configs['Hyper']['batch_size'],
                shuffle=True,
                num_workers=configs['Hyper']['num_workers'],
                collate_fn=Data_collate_poly, pin_memory=True)

        valset = Dataset_Inria(val_ROOT, mode='valid', N=NUM_POINTS, dilate=dilate_pixels)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=configs['Hyper']['batch_size'],
            shuffle=True,
            num_workers=configs['Hyper']['num_workers'],
            collate_fn=Data_collate_poly, pin_memory=True)

    if configs['Experiment']['dataset_name'] == 'CrowdAI':
        if not configs['Experiment']['evaluate']:
            trainset = Dataset_MC(train_ROOT, mode='train', N=NUM_POINTS, dilate=dilate_pixels)
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=configs['Hyper']['batch_size'],
                shuffle=True,
                num_workers=configs['Hyper']['num_workers'],
                collate_fn=Data_collate_poly, pin_memory=True)

        valset = Dataset_MC(val_ROOT, mode='valid', N=NUM_POINTS, dilate=dilate_pixels)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=configs['Hyper']['batch_size'],
            shuffle=True,
            num_workers=configs['Hyper']['num_workers'],
            collate_fn=Data_collate_poly, pin_memory=True)

    if configs['Experiment']['object_type'] == 'water':
        if not configs['Experiment']['evaluate']:
            trainset = Dataset_water(train_ROOT, mode='train', N=NUM_POINTS, dilate=dilate_pixels)
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=configs['Hyper']['batch_size'],
                shuffle=True,
                num_workers=configs['Hyper']['num_workers'],
                collate_fn=Data_collate_poly, pin_memory=True)

        validset = Dataset_water(val_ROOT, mode='valid', N=NUM_POINTS, dilate=dilate_pixels)
        val_loader = torch.utils.data.DataLoader(
            validset,
            batch_size=configs['Hyper']['batch_size'],
            shuffle=True,
            num_workers=configs['Hyper']['num_workers'],
            collate_fn=Data_collate_poly, pin_memory=True)

    if configs['Experiment']['dataset_name'] in ['Massachusetts', 'DeepGlobe']:
        if not configs['Experiment']['evaluate']:
            trainset = Dataset_road(train_ROOT, mode='train', N=NUM_POINTS, dilate=dilate_pixels)
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=configs['Hyper']['batch_size'],
                shuffle=True,
                num_workers=configs['Hyper']['num_workers'], collate_fn=Data_collate_road, pin_memory=True)

        valset = Dataset_road(val_ROOT, mode='valid', N=NUM_POINTS, dilate=dilate_pixels)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=configs['Hyper']['batch_size'],
            shuffle=True,
            num_workers=configs['Hyper']['num_workers'], collate_fn=Data_collate_road, pin_memory=True)

    return train_loader, val_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def Dataset_Loader_DDP(args):
    configs = args.configs
    train_ROOT = configs['Paths']['TrainRoot']
    val_ROOT = configs['Paths']['ValRoot']
    NUM_POINTS = configs['Model']['NUM_POINTS']
    dilate_pixels = configs['Model']['dilate_pixels']
    train_loader = []
    val_loader = []
    collate_fn = Data_collate_poly
    drop_last_flag = False if  configs['Experiment']['evaluate'] else True
    if configs['Experiment']['dataset_name'] in ['Inria', 'GID', 'GF', 'CrowdAI']:        
        if not configs['Experiment']['evaluate']:
            trainset = Dataset_Inria(train_ROOT, mode='train', N=NUM_POINTS, dilate=dilate_pixels)
        valset = Dataset_Inria(val_ROOT, mode='valid', N=NUM_POINTS, dilate=dilate_pixels)


    if configs['Experiment']['dataset_name'] in ['Massachusetts', 'DeepGlobe']:
        collate_fn = Data_collate_road
        if not configs['Experiment']['evaluate']:
            trainset = Dataset_road(train_ROOT, mode='train', N=NUM_POINTS, dilate=dilate_pixels)
        valset = Dataset_road(val_ROOT, mode='valid', N=NUM_POINTS, dilate=dilate_pixels)

    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=20)

    if not configs['Experiment']['evaluate']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=configs['Hyper']['batch_size'],
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=init_fn,
            sampler=train_sampler, collate_fn=collate_fn,
            drop_last=drop_last_flag)

    val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=configs['Hyper']['batch_size'],
        num_workers=args.workers,
        pin_memory=True, collate_fn=collate_fn,
        worker_init_fn=init_fn,
        sampler=val_sampler,
        drop_last=drop_last_flag)

    return train_loader, val_loader