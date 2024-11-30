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
    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float64) + 1e-9

    col_sum = np.sum(confu_mat, axis=1)
    raw_sum = np.sum(confu_mat, axis=0)

    '''OA'''
    oa = 0
    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()

    TP = []

    for i in range(class_num):
        TP.append(confu_mat[i, i])

    # f1-score
    TP = np.array(TP)
    FP = col_sum - TP
    FN = raw_sum - TP

    # precision，recall, IOU
    precision = TP / col_sum
    recall = TP / raw_sum
    f1 = 2 * (precision * recall) / (precision + recall)
    iou = TP / (TP + FP + FN)

    return oa, precision, recall, f1, iou



def performMetrics(pred, true, n_classes=2):
    pred[pred > 0] = 1
    true[true > 0] = 1
    confu_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    confu_matrix += get_confusion_matrix_with_counter(true, pred, class_num=n_classes)
    oa, precision, recall, f1, iou = metrics(confu_matrix)

    stats = {
        'Pixel Accuracy': oa * 100,
        'Precision': np.nanmean(precision) * 100,
        'Recall': np.nanmean(recall) * 100,
        'F1-score': np.nanmean(f1) * 100,
        'IoU': np.nanmean(iou) * 100
    }

    return stats

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
