import torch
import numpy as np
from scipy.spatial import cKDTree
from collections import Counter
import shapely
from skimage import measure
import cv2


def get_confusion_matrix_with_counter(label, predict, class_num=2):
    confu_list = []
    for i in range(class_num):
        c = Counter(label[np.where(predict == i)])
        single_row = []
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int64)

def metrics(confu_mat_total):
    '''
    confu_mat_total: 总的混淆矩阵
    keep_background：是否干掉背景
    '''

    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float64) + 1e-9

    col_sum = np.sum(confu_mat, axis=1)
    raw_sum = np.sum(confu_mat, axis=0)

    '''计算各类面积比，以求OA值'''
    oa = 0
    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()

    # 将混淆矩阵写入excel中
    TP = []  # 识别中每类分类正确的个数

    for i in range(class_num):
        TP.append(confu_mat[i, i])

    # 计算f1-score
    TP = np.array(TP)
    FP = col_sum - TP
    FN = raw_sum - TP

    # 计算并写出precision，recall, IOU
    precision = TP / col_sum
    recall = TP / raw_sum
    f1 = 2 * (precision * recall) / (precision + recall)
    iou = TP / (TP + FP + FN)

    return oa, precision, recall, f1, iou



def performMetrics(pred, true, n_classes=2):
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(true):
        true = true.cpu().numpy()
    pred[pred > 0] = 1
    true[true > 0] = 1
    confu_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(pred.shape[0]):
        confu_matrix += get_confusion_matrix_with_counter(true[i], pred[i], class_num=n_classes)
    oa, precision, recall, f1, iou = metrics(confu_matrix)

    stats = {
        'Pixel Accuracy': oa * 100,
        'Precision': np.nanmean(precision) * 100,
        'Recall': np.nanmean(recall) * 100,
        'F1-score': np.nanmean(f1) * 100,
        'IoU': np.nanmean(iou) * 100
    }

    return stats



def points_detection_acc(points_pred, points_true):
    if torch.is_tensor(points_pred):
        points_pred = points_pred.cpu().numpy()
    recall = {'recall2': [], 'recall5': [], 'recall10': []}
    precision = {'precision2': [], 'precision5': [], 'precision10': []}
    for b in range(points_pred.shape[0]):
        points_pred_tmp = points_pred[b]
        points_true_tmp = points_true[b]
        if len(points_true_tmp) == 0:
            continue
        if type(points_true_tmp) is not np.ndarray:
            points_true_tmp = np.concatenate(points_true_tmp,axis=0)
        pt_true_tree = cKDTree(points_true_tmp)
        pt_pred_tree = cKDTree(points_pred_tmp)

        num_pred = points_pred_tmp.shape[0]
        num_true = points_true_tmp.shape[0]
        true2 = 0
        true5 = 0
        true10 = 0
        pred2 = 0
        pred5 = 0
        pred10 = 0
        for pt_true in points_true_tmp:
            dis, idx = pt_pred_tree.query(pt_true)
            if dis <= 2:
                true2 += 1
            if dis <= 5:
                true5 += 1
            if dis <= 10:
                true10 += 1
        # for pt_pred in points_pred_tmp:
        #     dis, idx = pt_true_tree.query(pt_pred)
        #     if dis <= 2:
        #         pred2 += 1
        #     if dis <= 5:
        #         pred5 += 1
        #     if dis <= 10:
        #         pred10 += 1
        recall['recall2'].append(true2 / num_true)
        recall['recall5'].append(true5 / num_true)
        recall['recall10'].append(true10 / num_true)
        # precision['precision2'].append(pred2 / num_pred)
        # precision['precision5'].append(pred5 / num_pred)
        # precision['precision10'].append(pred10 / num_pred)

    if len(recall['recall2']):
        recall['recall2'] = sum(recall['recall2']) / len(recall['recall2'])
        recall['recall5'] = sum(recall['recall5']) / len(recall['recall5'])
        recall['recall10'] = sum(recall['recall10']) / len(recall['recall10'])
        # precision['precision2'] = sum(precision['precision2']) / len(precision['precision2'])
        # precision['precision5'] = sum(precision['precision5']) / len(precision['precision5'])
        # precision['precision10'] = sum(precision['precision10']) / len(precision['precision10'])
    return recall, precision


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def update_acc(recall, precision, b_recall, b_precision):
    for k, v in recall.items():
        if isinstance(b_recall[k], list):
            continue
        recall[k].append(b_recall[k])
    # for k, v in precision.items():
    #     if isinstance(b_precision[k], list):
    #         continue
    #     precision[k].append(b_precision[k])
    return recall, precision


def update_stats(stats, stats_batch, key='Mask'):
    for k, v in stats[key].items():
        stats[key][k].append(stats_batch[k])

    return stats


def summary_stats(stats):
    for k, v in stats.items():
        print('------', k, '------')
        for key, value in stats[k].items():
            assert isinstance(value, list)
            value = [i for i in value if isinstance(i, float)]

            print(str(key), ':', str(np.nanmean(value)))
