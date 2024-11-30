import random
import numpy as np
import os
import torch
import errno
from collections import OrderedDict
from jsmin import jsmin
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    model_dir = os.path.join(record_root, 'weights')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir, record_root


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
    for k, v in pre_weight.items():
        t_n += 1
        if k in new_model_dict:
            k = 'module.' + k if 'module' not in k else k
            v_n += 1
            pretrained_dict[k] = v
            # print(k)
    # os._exit()
    print(f'{v_n}/{t_n} weights have been loaded!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def Data_collate(batch):
    # variables as tensor
    ori_imgs = []
    mask = []
    boundary_mask = []
    names = []
    for im in batch:
        ori_imgs.append(torch.from_numpy(im['ori_img']))
        mask.append(torch.from_numpy(im['mask']).unsqueeze(0))
        boundary_mask.append(torch.from_numpy(im['boundary_mask']).unsqueeze(0))
        names.append(im['name'])

    ori_img_collection = torch.stack(ori_imgs, dim=0)
    mask_collection = torch.stack(mask, dim=0)
    boudary_mask_collection = torch.stack(boundary_mask, dim=0)

    sample = {'ori_images': ori_img_collection,
              'masks': mask_collection, 'boundary_masks': boudary_mask_collection, 'names': names}
    return sample

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def load_weight(args):
    if args.configs['Experiment']['detection_resume']:
        print('Loading Detection Model......')
        detection_weight_path = os.path.join(args.model_dir, args.configs['Paths']['pretrained_detection_weight_name'])
        # detection_weight_path = '/data01/ybn/Projects/UAV/cvpr/records/Inria_dff_320pt/weights/points_best.pkl'
        args.detection_model = load_ckpt(detection_weight_path, args.detection_model)

    if args.configs['Experiment']['match_resume']:
        print('Loading Graph Model......')
        match_weight_path = os.path.join(args.model_dir, args.configs['Paths']['pretrained_match_weight_name'])
        args.match_model = load_ckpt(match_weight_path, args.match_model)
    return args



