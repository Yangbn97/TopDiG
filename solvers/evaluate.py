import cv2
from pycocotools.coco import COCO
from skimage.morphology import skeletonize
from skimage.draw import polygon2mask
from skimage import io
from utils.poly_utils import *
from utils.setting_utils import *
from utils.metric_utils import *

def coco2mask(anno_file, img_path):
    coco = COCO(anno_file)
    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['building'])
    img_list = coco.getImgIds(catIds=catIds)
    mask_list = []
    for index in range(len(img_list)):
        imgId = img_list[index]
        img_dic = coco.loadImgs(imgId)[0]

        file_name = img_dic['file_name']

        file_path = os.path.join(img_path, file_name)
        annIds = coco.getAnnIds(imgIds=img_dic['id'], catIds=catIds, iscrowd=None)
        ann = coco.loadAnns(annIds)

        img = io.imread(file_path)

        instance_polys = [np.array(poly).reshape(-1, 2) for obj in ann for poly in obj['segmentation']]
        polys = [cnt[:, [-1, 0]] for cnt in instance_polys]
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for i in range(len(polys)):
            mask += polygon2mask((img.shape[0], img.shape[1]), polys[i])
        mask = np.uint8(mask) * 255

        mask_list.append(mask)

    return mask_list

def calculate_metrics(args):
    if args.configs['Experiment']['object_type'] == 'line':
        stats = {'Topology-wise': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}}
    else:
        stats = {'Pixel-wise': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []},
                 'Topology-wise': {'Pixel Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'IoU': []}}
    label_dir = args.configs['Paths']['TestLabelRoot']
    seg_out_dir = os.path.join(args.save_root, 'seg')
    if not os.path.exists(seg_out_dir):
        seg_out_dir = os.path.join(args.save_root, args.model_name)

    seg_list = os.listdir(seg_out_dir)
    seg_true_list = []
    # if args.configs['Experiment']['dataset_name'] == 'CrowdAI':
    #     seg_true_list = coco2mask(args.configs['Paths']['TestLabelRoot'], args.configs['Paths']['TestRoot'])

    for file_i in tqdm(range(len(seg_list))):
        filename = seg_list[file_i]
        seg_pred_path = os.path.join(seg_out_dir, filename)
        seg_pred = cv2.imread(seg_pred_path, 0)
        seg_pred[seg_pred > 0] = 1
        # if args.configs['Experiment']['dataset_name'] == 'CrowdAI':
        #     seg_true = seg_true_list[file_i]
        # else:
        label_name = filename
        if args.configs['Experiment']['dataset_name'] == 'DeepGlobe':
            label_name = label_name.replace('.tif','.png')
        seg_true_path = os.path.join(label_dir, label_name)   
        seg_true = cv2.imread(seg_true_path, 0)
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



