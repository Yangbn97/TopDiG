import os

import matplotlib.pyplot as plt
import sys

from torch.cuda.amp import autocast

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
from datetime import datetime
import warnings
import cv2
from skimage import io
from shapely.geometry import MultiPolygon, Polygon
from models.Graph import *
from utils.poly_utils import *
from utils.metric_utils import *
from utils.solver_utils import *
from utils.save_utils import line2raster, poly2raster
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_inference(args):
    setup_seed(20)
    TestRoot = args.configs['Paths']['TestRoot']
    img_dir = TestRoot

    infer_time = AverageMeter()
    model_time = AverageMeter()

    img_list = os.listdir(img_dir)
    args.detection_model.eval()
    args.match_model.eval()

    for filename in tqdm(img_list):
        start = time.time()
        basename = filename.split('.')[0]
        args.filename = basename
        img = io.imread(os.path.join(img_dir, filename))

        args.shp_dir = os.path.join(args.save_root, 'shp')
        args.raster_dir = os.path.join(args.save_root, 'seg')
        os.makedirs(args.shp_dir, exist_ok=True)
        os.makedirs(args.raster_dir, exist_ok=True)
        shp_save_path = os.path.join(args.shp_dir, basename + '.shp')
        seg_save_path = os.path.join(args.raster_dir, basename + '.tif')
        if args.configs['Experiment']['object_type'] == 'line':
            FullDiGraph, subgraphs, model_time_tmp = predict_line_graph_WithOverlap_simple(args, img,
                                                                                           patch_size=args.configs['Model']['input_img_size'],
                                                                                           intersect_area=0)
            if args.eval or args.save_seg:
                line2raster(img, seg_save_path, FullDiGraph)
            if args.save_shp:
                LineGraph2shp(FullDiGraph, shp_save_path, os.path.join(img_dir, filename))
        else:
            shapely_polygons, model_time_tmp = predict_poly_graph_WithOverlap_simple(args, img, patch_size=args.configs['Model']['input_img_size'],
                                                                                     intersect_area=100)
            if isinstance(shapely_polygons, MultiPolygon):
                polys = [np.asarray(py.exterior.coords) for py in shapely_polygons]
            elif isinstance(shapely_polygons, Polygon):
                polys = [np.asarray(shapely_polygons.exterior.coords)]
            else:
                polys = shapely_polygons
            if args.eval or args.save_seg:
                poly2raster(img, seg_save_path, polys)
            if args.save_shp:
                poly2shp(os.path.join(img_dir, filename), shp_save_path, polys)

        model_time.update(model_time_tmp)
        infer_time.update((time.time() - start))
    print('Average model times is {} second per image'.format(model_time.avg))
    print('Average inference times is {} second per image'.format(infer_time.avg))
    return args

