import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from solvers.evaluate import *
from solvers.inference import *
from models.TCND import *
from models.Graph import *
from utils.setting_utils import *
from utils.summary import make_print_to_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')

setup_seed(20)
parser = argparse.ArgumentParser(description='Params for label making')
parser.add_argument(
    '-c', '--configs_path',
    # type=str,
    # default=r'./configs/config_GID.json',
    default=r'./configs/config_Massachusetts.json',
    # default=r'./configs/config_Inria.json',
    help='Path of the configs file.')

parser.add_argument(
    '--eval',
    help='Command to infer raster outputs and then calculate metrics')

parser.add_argument(
    '--eval_only',
    help='Only calculate metrics')

parser.add_argument(
    '--save_shp',
    help='Command to infer and save results as shapefile')

parser.add_argument(
    '--save_seg',
    help='Command to only infer and save results as raster map')



args = parser.parse_args()
assert args.configs_path is not None, "Argument --configs must be specified. Run 'python main.py --help' for help on arguments."
configs = load_config(args.configs_path)
args.configs = configs
args.save_shp = args.configs['Experiment']['save_shp']
args.save_seg = args.configs['Experiment']['save_seg']
args.eval = args.configs['Experiment']['evaluate']
args.eval_only = args.configs['Experiment']['eval_only']

args.gap = args.configs['Model']['phi']
args.save_root = args.configs['Paths']['SaveRoot']

args.model_dir, args.record_root = build_roots(configs)


make_print_to_file(args)

args.detection_model = get_TCND(backbone='resnet50', pretrained=True)
feature_dim = 64
args.match_model = Graph_Generator(sinkhorn=args.configs['Model']['Sinkhorn'], featuremap_dim=feature_dim, configs=args.configs)

args.detection_model = nn.DataParallel(args.detection_model)
args.detection_model = args.detection_model.to(device)

args.match_model = nn.DataParallel(args.match_model)
args.match_model = args.match_model.to(device)


args = load_weight(args)


def main():
    print(configs)
    if args.eval_only:
        print('Calculating metrics......')
        calculate_metrics(args)
        return
    print('Running inference.....')
    run_inference(args)
    if args.eval:
        print('Calculating metrics......')
        calculate_metrics(args)
        return

if __name__ == '__main__':
    main()
