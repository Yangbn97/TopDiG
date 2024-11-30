import cv2
from skimage.morphology import skeletonize
from utils.poly_utils import *
from utils.setting_utils import *
from utils.metric_utils import *

def calculate_metrics(args):
    if args.configs['Experiment']['object_type'] == 'line':
        stats = {'Topology-wise': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}}
    else:
        stats = {'Pixel-wise': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []},
                 'Topology-wise': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}}
    label_dir = args.configs['Paths']['TestLabelRoot']
    seg_out_dir = os.path.join(args.save_root, 'seg')

    seg_list = os.listdir(seg_out_dir)

    for filename in tqdm(seg_list):
        seg_pred_path = os.path.join(seg_out_dir, filename)
        label_name = filename.replace('.tif','_label.tif') if args.configs['Experiment']['dataset_name'] == 'GID' else filename
        seg_true_path = os.path.join(label_dir, label_name)
        seg_pred = cv2.imread(seg_pred_path, 0)
        seg_true = cv2.imread(seg_true_path, 0)
        seg_pred[seg_pred > 0] = 1
        seg_true[seg_true > 0] = 1

        if args.configs['Experiment']['object_type'] == 'line':
            skel_true = np.uint8(skeletonize(seg_true) * 255)
            center_true = cv2.GaussianBlur(skel_true, ksize=(args.configs['Model']['dilate_pixels'], args.configs['Model']['dilate_pixels']), sigmaX=1,
                                                 sigmaY=1)
            center_true[center_true > 0] = 1
            center_true = np.uint8(center_true)

            center_pred = cv2.GaussianBlur(seg_pred*255, ksize=(args.configs['Model']['dilate_pixels'], args.configs['Model']['dilate_pixels']), sigmaX=1,
                                                 sigmaY=1)
            center_pred[center_pred > 0] = 1
            center_pred = np.uint8(center_pred)

            stats_center = performMetrics(center_pred, center_true)
            stats = update_stats(stats, stats_center, key='Topology-wise')

        else:
            boundary_pred = region2boundary(seg_pred)
            boundary_true = region2boundary(seg_true)
            boundary_mask_pred = cv2.GaussianBlur(boundary_pred, ksize=(args.configs['Model']['dilate_pixels'], args.configs['Model']['dilate_pixels']), sigmaX=1,
                                             sigmaY=1)
            boundary_mask_true = cv2.GaussianBlur(boundary_true, ksize=(args.configs['Model']['dilate_pixels'], args.configs['Model']['dilate_pixels']), sigmaX=1,
                                             sigmaY=1)
            boundary_mask_pred[boundary_mask_pred > 0] = 1
            boundary_mask_true[boundary_mask_true > 0] = 1
            boundary_mask_pred = np.uint8(boundary_mask_pred)
            boundary_mask_true = np.uint8(boundary_mask_true)

            stats_mask = performMetrics(seg_pred, seg_true)
            stats_boundary = performMetrics(boundary_mask_pred,boundary_mask_true)

            stats = update_stats(stats, stats_mask, key='Pixel-wise')
            stats = update_stats(stats, stats_boundary, key='Topology-wise')


    summary_stats(stats)



