import random
import numpy as np
import os
import torch
import torch.nn as nn
import errno
from collections import OrderedDict
from jsmin import jsmin
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_ddp(args, cfg, key='DDP'):
    args.dist_url = cfg[key]['dist_url']
    args.dist_backend = cfg[key]['dist_backend']
    args.multiprocessing_distributed = cfg[key]['multiprocessing_distributed']
    args.world_size = cfg[key]['world_size']
    args.rank = cfg[key]['rank']
    args.sync_bn = cfg[key]['sync_bn']
    return args

def to_gpu(args, model, mode='DP'):
    if mode == 'DP':
          model = nn.DataParallel(model)
          model = model.cuda()
    elif mode == 'DDP':
        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[args.gpu],
                                                find_unused_parameters=True)
    else:
        pass

    return model


def mkdirs(newdir):
    """
    make directory with parent path
    :param newdir: target path
    """
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


def load_config(config_name="configs", config_dirpath="", try_default=False):
    if os.path.splitext(config_name)[1] == ".json":
        config_filepath = os.path.join(config_dirpath, config_name)
    else:
        config_filepath = os.path.join(config_dirpath, config_name + ".json")

    try:
        with open(config_filepath, 'r') as f:
            minified = jsmin(f.read())
            # config = json.loads(minified)
            try:
                config = json.loads(minified)
            except json.decoder.JSONDecodeError as e:
                print("ERROR: Parsing configs failed:")
                print(e)
                print("Minified JSON causing the problem:")
                print(str(minified))
                exit()
        return config
    except FileNotFoundError:
        if config_name == "configs" and config_dirpath == "":
            print(
                "WARNING: the default configs file was not found....")
            return None
        elif try_default:
            print(
                "WARNING: configs file {} was not found, opening default configs file configs.defaults.json instead.".format(
                    config_filepath))
            return load_config()
        else:
            print(
                "WARNING: configs file {} was not found.".format(config_filepath))
            return None


def build_roots(configs):
    record_root = os.path.join('./records/', configs['Paths']['records_filename'])
    os.makedirs(record_root, exist_ok=True)
    work_dir = '{}/{}'.format(os.path.join(record_root, 'runs'), 'Train')
    os.makedirs(work_dir, exist_ok=True)
    vis_root = os.path.join(record_root, 'vis')
    os.makedirs(vis_root, exist_ok=True)
    model_dir = os.path.join(record_root, 'weights')
    os.makedirs(model_dir, exist_ok=True)
    return work_dir, vis_root, model_dir, record_root


def load_model(weight_dir, model):
    pretext_model = torch.load(weight_dir)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pretext_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model


def load_ckpt(weight_dir, model):
    pre_weight = torch.load(weight_dir)
    new_pre_weight = OrderedDict()
    # pre_weight =torch.jit.load(resume)
    model_dict = model.state_dict()
    new_model_dict = OrderedDict()

    for k, v in pre_weight.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        new_pre_weight[new_k] = v
    for k, v in model_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        new_model_dict[new_k] = v

    pre_weight = new_pre_weight  # ["model_state"]
    pretrained_dict = {}
    t_n = 0
    v_n = 0
    unmatched_keys = []
    for k, v in pre_weight.items():
        t_n += 1
        if k in new_model_dict and v.shape == new_model_dict[k].shape:
            # k = 'module.' + k if 'module' not in k else k
            v_n += 1
            pretrained_dict[k] = v
            # print(k)
        else:
            unmatched_keys.append(k)
    # os._exit()
    print(f'{v_n}/{t_n} weights have been loaded!')
    print('Unmatched keys in state_dict:', unmatched_keys)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

def load_swin_b_ckpt_for_TCSwin(weight_dir, model, load_range=[3,-43]):
    ckpt1 = torch.load(weight_dir)
    ckpt2 = model.state_dict()
    kl1 = list(ckpt1.keys())
    for i, k in enumerate(list(ckpt2.keys())[load_range[0]:load_range[1]]):
        ckpt2[k] = ckpt1[kl1[i]]
    msg = model.load_state_dict(ckpt2, strict=False)
    print(f'Load swin_transformer: {msg}')

    return model

def load_swin_b_ckpt(weight_dir, model, load_key='backbone'):
    pre_weight = torch.load(weight_dir)
    model_dict = model.state_dict()
    pre_weight_keys = list(pre_weight.keys())
    backbone_keys = [k for k in model_dict if load_key in k]
    for i, k in enumerate(backbone_keys):
        model_dict[k] = pre_weight[pre_weight_keys[i]]
    msg = model.load_state_dict(model_dict, strict=False)
    print(f'Load swin_transformer: {msg}')

    return model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_weight(args):
    if args.configs['Experiment']['detection_resume']:
        print('Loading Detection Model......')
        detection_weight_path = args.configs['Paths']['pretrained_detection_weight_path']
        # detection_weight_path = '/data02/ybn/Projects/TopDiG/records/Massachusetts/weights/points_best.pkl'
        args.detection_model = load_ckpt(detection_weight_path, args.detection_model)
    if not args.configs['Experiment']['pretrain']:

        print('Loading pretained ViT......')
        ViT_weight_path = args.configs['Paths']['pretrained_ViT_weight_path']
        args.match_model = load_ckpt(ViT_weight_path, args.match_model)

        if args.configs['Experiment']['match_resume']:
            print('Loading Graph Model......')
            match_weight_path = args.configs['Paths']['pretrained_match_weight_path']
            # match_weight_path = '/data02/ybn/Projects/TopDiG/records/Massachusetts/weights/connection_best.pkl'
            args.match_model = load_ckpt(match_weight_path, args.match_model)
    return args

def save_model(self, best_loss, current_loss, epoch):
    best_loss['total_loss'] = current_loss[0].avg
    best_loss['point_loss'] = current_loss[1].avg
    best_loss['match_loss'] = current_loss[2].avg
    best_loss['boundary IoU'] = current_loss[3].avg
    best_loss['mask IoU'] = current_loss[4].avg
    if self.configs['Experiment']['detection_model_save']:
        path_detection = os.path.join(self.model_dir,
                                      'points_epoch{}_valLoss_{:.5f}_IoU_{:.4f}_ptloss_{:.4f}.pkl'.format(epoch, best_loss[
                                          'total_loss'], best_loss['mask IoU'],best_loss['point_loss']))
        torch.save(self.detection_model.state_dict(), path_detection)
        print('Saving Detection Model=>', path_detection)
    if self.configs['Experiment']['match_model_save']:
        path_match = os.path.join(self.model_dir,
                                  'connection_epoch{}_valLoss_{:.5f}_IoU_{:.4f}_matchloss_{:.5f}.pkl'.format(epoch,
                                                                                                  best_loss[
                                                                                                      'total_loss'],
                                                                                                  best_loss[
                                                                                                      'mask IoU'],
                                                                                                  best_loss[
                                                                                                      'match_loss']))
        torch.save(self.match_model.state_dict(), path_match)
        print('Saving Graph Model=>', path_match)

    return best_loss

def save_pretrain_model(args, best_loss, epoch):
    # best_iou = ious_mask.avg
    if args.configs['Experiment']['detection_model_save']:
        path_detection = os.path.join(args.model_dir,
                                      'Pretrain_Epoch{}_valLoss_{:.4f}.pkl'.format(epoch, best_loss['total_loss']))
        torch.save(args.detection_model.state_dict(), path_detection)
        print('Saving Model to =>', path_detection)



