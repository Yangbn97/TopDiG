import matplotlib
import networkx as nx
import scipy
import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from skimage.draw import polygon2mask
from utils.poly_utils import scores_to_permutations, AdjMatrix2DiGraph
from utils.NMS import gpu_NMS


def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]


def visualize_output_poly(batch, batch_out, epoch, vis_dir=None, mode='train',title=None):
    heat_dir = os.path.join(vis_dir, 'heat')
    os.makedirs(heat_dir, exist_ok=True)
    vis_dir = os.path.join(vis_dir, mode)
    os.makedirs(vis_dir, exist_ok=True)


    heatmap = batch['heatmaps']
    ori_img = batch['ori_images']
    names = batch['names']
    polys_gt = batch['polys']
    PMs = batch['PM_label']
    heatmap_pred = F.sigmoid(batch_out['heatmap_pred'])
    ori_points_pred = batch_out['points_pred'].cpu().numpy()
    PM_pred = torch.exp(batch_out['PM_pred'])

    PM_pred = scores_to_permutations(PM_pred)
    BN = min(5, heatmap.shape[0])
    for n in range(0, BN):
        heatmap_gt = heatmap.squeeze(1).detach().cpu().numpy()[n, ...]
        heatmap_predict = heatmap_pred.squeeze(1).detach().cpu().numpy()[n, ...]

        if np.sum(heatmap_gt) == 0:
            continue

        points_true = polys_gt[n]
        points_pred = ori_points_pred[n]

        points_pred = np.asarray(points_pred)
        # points_true = np.asarray(points_true)

        PM_true = PMs.squeeze(1).cpu().detach().numpy()[n, ...]
        adj_index_pred = PM_pred.cpu().detach().numpy()[n, ...]

        DiGraph_pred = AdjMatrix2DiGraph(adj_index_pred, points_pred)
        circles_pred = nx.simple_cycles(DiGraph_pred)
        polys_pred = []
        for list in circles_pred:
            if len(list) > 1:
                poly_tmp_pred = [points_pred[i] for i in list]
                polys_pred.append(poly_tmp_pred)

        image = ori_img.numpy()[n, ...]
        img = image.astype(np.uint8)
        img_point_pred = img.copy()
        img_point_ture = img.copy()
        img_PM_pred = img.copy()
        img_PM_true = img.copy()

        for cnt in polys_pred:
            if len(cnt) <= 3:
                continue
            for p in range(len(cnt)):
                if p == len(cnt) - 1:
                    # cv2.circle(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), 4, (255, 0, 0), -1)
                    # cv2.circle(img_PM_pred, (int(cnt[0][1]), int(cnt[0][0])), 4, (255, 0, 0), -1)
                    # cv2.arrowedLine(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[0][1]), int(cnt[0][0])),
                    #                 (242, 203, 5), thickness=2, tipLength=0.3)
                    mid = (cnt[p] + cnt[0]) / 2
                    cv2.line(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[0][1]), int(cnt[0][0])),
                             (242, 203, 5), 2)
                    cv2.arrowedLine(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(mid[1]), int(mid[0])),
                                    (242, 203, 5), thickness=2, tipLength=0.3)
                    cv2.circle(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), 4, (5, 203, 242), -1)
                    cv2.circle(img_PM_pred, (int(cnt[0][1]), int(cnt[0][0])), 4, (5, 203, 242), -1)
                else:
                    # cv2.circle(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), 4, (255, 0, 0), -1)
                    # cv2.circle(img_PM_pred, (int(cnt[p + 1][1]), int(cnt[p + 1][0])), 4, (255, 0, 0), -1)
                    # cv2.arrowedLine(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])),
                    #                 (int(cnt[p + 1][1]), int(cnt[p + 1][0])),
                    #                 (242, 203, 5), thickness=2, tipLength=0.3)
                    mid = (cnt[p] + cnt[p + 1]) / 2
                    cv2.line(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[p + 1][1]), int(cnt[p + 1][0])),
                             (242, 203, 5), thickness=2)
                    cv2.arrowedLine(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(mid[1]), int(mid[0])),
                                    (242, 203, 5), thickness=2, tipLength=0.3)
                    cv2.circle(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), 4, (5, 203, 242), -1)
                    cv2.circle(img_PM_pred, (int(cnt[p + 1][1]), int(cnt[p + 1][0])), 4, (5, 203, 242), -1)

        for cnt in points_true:
            for i in range(len(cnt) - 1):
                pt = cnt[i]
                next = cnt[i + 1]
                mid = (cnt[i] + cnt[i + 1]) / 2
                cv2.line(img_point_ture, (int(pt[1]), int(pt[0])), (int(next[1]), int(next[0])),
                         (242, 203, 5), thickness=2)
                cv2.arrowedLine(img_point_ture, (int(pt[1]), int(pt[0])), (int(mid[1]), int(mid[0])), (242, 203, 5),
                                thickness=2, tipLength=0.3)
                cv2.circle(img_point_ture, (int(pt[1]), int(pt[0])), 4, (5, 203, 242), -1)
                if i != len(cnt) - 2:
                    cv2.circle(img_point_ture, (int(next[1]), int(next[0])), 4, (5, 203, 242), -1)

        pair_idx = np.argwhere(PM_true == 1)
        pair_idx = [p for p in pair_idx if p[0] != p[1]]
        for pair in pair_idx:
            pt = points_pred[pair[0]]
            adj_pt = points_pred[pair[1]]
            cv2.circle(img_PM_true, (int(pt[1]), int(pt[0])), 4, (255, 0, 0), -1)
            cv2.circle(img_PM_true, (int(adj_pt[1]), int(adj_pt[0])), 4, (255, 0, 0), -1)
            cv2.arrowedLine(img_PM_true, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])),
                            (242, 203, 5), thickness=2, tipLength=0.3)

        norm_img = np.zeros(heatmap_predict.shape)
        heatmap_predict = heatmap_predict.astype(np.float32)
        norm_img = cv2.normalize(heatmap_predict, norm_img, 0, 255, cv2.NORM_MINMAX)
        norm_img = np.asarray(norm_img, dtype=np.uint8)

        norm_img = cv2.applyColorMap(norm_img,cv2.COLORMAP_JET)
        # norm_img = cv2.cvtColor(norm_img,cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image[:,:,[2,1,0]])
        img_add = cv2.addWeighted(image,0.3,norm_img,0.7,0)
        # heat_save_path = os.path.join(heat_dir, '{}.png'.format(names[n]))
        # cv2.imwrite(heat_save_path,img_add)

        plt.figure()
        plt.subplot(231)
        plt.title('Points pred')
        plt.imshow(img_point_pred)
        plt.axis('off')
        plt.plot(points_pred[:, 1], points_pred[:, 0], 'o', markersize=1, color='r')
        plt.subplot(232)
        plt.title('GT')
        plt.imshow(img_point_ture)
        plt.axis('off')
        plt.subplot(234)
        plt.title('Connectivity pred')
        plt.imshow(img_PM_pred)
        plt.axis('off')
        plt.subplot(235)
        plt.title('Connectivity label')
        plt.imshow(img_PM_true)
        plt.axis('off')
        plt.subplot(233)
        plt.title('Heatmap gt')
        plt.imshow(heatmap_gt)
        plt.axis('off')
        plt.subplot(236)
        plt.title('Heatmap pred')
        plt.imshow(img_add)
        plt.axis('off')
        img_save_path = os.path.join(vis_dir, 'E{}_{}.png'.format(epoch, names[n]))
        if title is not None:
            plt.suptitle(title)
        plt.savefig(img_save_path, dpi=400, bbox_inches='tight')
        # plt.show()
        del points_true, points_pred, img_point_pred, img_point_ture, img_PM_true, img_PM_pred

def visualize_poly_eval(batch, batch_out, vis_dir=None, mode='train',title=None):
    vis_dir = os.path.join(vis_dir, mode)
    os.makedirs(vis_dir, exist_ok=True)
    show_dir = os.path.join(vis_dir, 'show')
    os.makedirs(show_dir, exist_ok=True)
    heat_dir = os.path.join(vis_dir, 'heat')
    os.makedirs(heat_dir, exist_ok=True)
    infer_dir = os.path.join(vis_dir, 'infer')
    os.makedirs(infer_dir, exist_ok=True)


    heatmap = batch['heatmaps']
    ori_img = batch['ori_images']
    names = batch['names']
    polys_gt = batch['polys']
    PMs = batch['PM_label']
    heatmap_pred = F.sigmoid(batch_out['heatmap_pred'])
    ori_points_pred = batch_out['points_pred'].cpu().numpy()
    PM_pred = torch.exp(batch_out['PM_pred'])

    PM_pred = scores_to_permutations(PM_pred)
    BN = ori_img.shape[0]
    for n in range(0, BN):
        heatmap_gt = heatmap.squeeze(1).detach().cpu().numpy()[n, ...]
        heatmap_predict = heatmap_pred.squeeze(1).detach().cpu().numpy()[n, ...]

        image = ori_img.numpy()[n, ...]
        img = image.astype(np.uint8)
        
        norm_img = np.zeros(heatmap_predict.shape)
        heatmap_predict = heatmap_predict.astype(np.float32)
        norm_img = cv2.normalize(heatmap_predict, norm_img, 0, 255, cv2.NORM_MINMAX)
        norm_img = np.asarray(norm_img, dtype=np.uint8)

        norm_img = cv2.applyColorMap(norm_img,cv2.COLORMAP_JET)
        # norm_img = cv2.cvtColor(norm_img,cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image[:,:,[2,1,0]])
        img_add = cv2.addWeighted(image,0.3,norm_img,0.7,0)
        
        heat_save_path = os.path.join(heat_dir, '{}.png'.format(names[n]))
        cv2.imwrite(heat_save_path, img_add)

        if np.sum(heatmap_gt) == 0:
            continue

        points_true = polys_gt[n]
        points_pred = ori_points_pred[n]

        points_pred = np.asarray(points_pred)
        # points_true = np.asarray(points_true)

        PM_true = PMs.squeeze(1).cpu().detach().numpy()[n, ...]
        adj_index_pred = PM_pred.cpu().detach().numpy()[n, ...]

        DiGraph_pred = AdjMatrix2DiGraph(adj_index_pred, points_pred)
        circles_pred = nx.simple_cycles(DiGraph_pred)
        polys_pred = []
        for list in circles_pred:
            if len(list) > 1:
                poly_tmp_pred = [points_pred[i] for i in list]
                polys_pred.append(poly_tmp_pred)

        img_point_pred = img.copy()
        img_point_ture = img.copy()
        img_PM_pred = img.copy()
        img_PM_pred = cv2.cvtColor(img_PM_pred,cv2.COLOR_RGB2BGR)
        img_PM_true = img.copy()

        for cnt in polys_pred:
            if len(cnt) <= 3:
                continue
            for p in range(len(cnt)):
                if p == len(cnt) - 1:
                    mid = (cnt[p] + cnt[0]) / 2
                    # cv2.line(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[0][1]), int(cnt[0][0])),
                    #          (242, 203, 5), 2)
                    # cv2.arrowedLine(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(mid[1]), int(mid[0])),
                    #                 (242, 203, 5), thickness=2, tipLength=0.3)
                    # cv2.circle(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), 4, (5, 203, 242), -1)
                    # cv2.circle(img_PM_pred, (int(cnt[0][1]), int(cnt[0][0])), 4, (5, 203, 242), -1)

                    cv2.line(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[0][1]), int(cnt[0][0])),
                             (5, 203, 242), 2)
                    cv2.arrowedLine(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(mid[1]), int(mid[0])),
                                    (5, 203, 242), thickness=2, tipLength=0.3)
                    cv2.circle(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), 4, (242, 203, 5), -1)
                    cv2.circle(img_PM_pred, (int(cnt[0][1]), int(cnt[0][0])), 4, (242, 203, 5), -1)
                else:
                    mid = (cnt[p] + cnt[p + 1]) / 2
                    # cv2.line(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[p + 1][1]), int(cnt[p + 1][0])),
                    #          (242, 203, 5), thickness=2)
                    # cv2.arrowedLine(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(mid[1]), int(mid[0])),
                    #                 (242, 203, 5), thickness=2, tipLength=0.3)
                    # cv2.circle(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), 4, (5, 203, 242), -1)
                    # cv2.circle(img_PM_pred, (int(cnt[p + 1][1]), int(cnt[p + 1][0])), 4, (5, 203, 242), -1)

                    cv2.line(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[p + 1][1]), int(cnt[p + 1][0])),
                             (5, 203, 242), thickness=2)
                    cv2.arrowedLine(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(mid[1]), int(mid[0])),
                                    (5, 203, 242), thickness=2, tipLength=0.3)
                    cv2.circle(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), 4, (242, 203, 5), -1)
                    cv2.circle(img_PM_pred, (int(cnt[p + 1][1]), int(cnt[p + 1][0])), 4, (242, 203, 5), -1)

        for cnt in points_true:
            for i in range(len(cnt) - 1):
                pt = cnt[i]
                next = cnt[i + 1]
                mid = (cnt[i] + cnt[i + 1]) / 2
                cv2.line(img_point_ture, (int(pt[1]), int(pt[0])), (int(next[1]), int(next[0])),
                         (242, 203, 5), thickness=2)
                cv2.arrowedLine(img_point_ture, (int(pt[1]), int(pt[0])), (int(mid[1]), int(mid[0])), (242, 203, 5),
                                thickness=2, tipLength=0.3)
                cv2.circle(img_point_ture, (int(pt[1]), int(pt[0])), 4, (5, 203, 242), -1)
                if i != len(cnt) - 2:
                    cv2.circle(img_point_ture, (int(next[1]), int(next[0])), 4, (5, 203, 242), -1)

        pair_idx = np.argwhere(PM_true == 1)
        pair_idx = [p for p in pair_idx if p[0] != p[1]]
        for pair in pair_idx:
            pt = points_pred[pair[0]]
            adj_pt = points_pred[pair[1]]
            cv2.circle(img_PM_true, (int(pt[1]), int(pt[0])), 4, (255, 0, 0), -1)
            cv2.circle(img_PM_true, (int(adj_pt[1]), int(adj_pt[0])), 4, (255, 0, 0), -1)
            cv2.arrowedLine(img_PM_true, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])),
                            (242, 203, 5), thickness=2, tipLength=0.3)

        infer_save_path = os.path.join(infer_dir, '{}.png'.format(names[n]))
        cv2.imwrite(infer_save_path, img_PM_pred)

        plt.figure()
        plt.subplot(231)
        plt.title('Points pred')
        plt.imshow(img_point_pred)
        plt.axis('off')
        plt.plot(points_pred[:, 1], points_pred[:, 0], 'o', markersize=1, color='r')
        plt.subplot(232)
        plt.title('GT')
        plt.imshow(img_point_ture)
        plt.axis('off')
        plt.subplot(234)
        plt.title('Connectivity pred')
        img_PM_pred = cv2.cvtColor(img_PM_pred,cv2.COLOR_RGB2BGR)
        plt.imshow(img_PM_pred)
        plt.axis('off')
        plt.subplot(235)
        plt.title('Connectivity label')
        plt.imshow(img_PM_true)
        plt.axis('off')
        plt.subplot(233)
        plt.title('Heatmap gt')
        plt.imshow(heatmap_gt)
        plt.axis('off')
        plt.subplot(236)
        plt.title('Heatmap pred')
        plt.imshow(img_add)
        plt.axis('off')
        img_save_path = os.path.join(show_dir, '{}.png'.format(names[n]))
        plt.savefig(img_save_path, dpi=400, bbox_inches='tight')
        # plt.show()
        del points_true, points_pred, img_point_pred, img_point_ture, img_PM_true, img_PM_pred

def visualize_output_road(batch, batch_out, epoch, vis_dir=None, mode='train',
                                 threshold=0.45, score=0.6):
    heat_dir = os.path.join(vis_dir, 'heat')
    os.makedirs(heat_dir, exist_ok=True)
    vis_dir = os.path.join(vis_dir, mode)
    os.makedirs(vis_dir, exist_ok=True)

    thr = threshold
    heatmap = batch['heatmaps']
    ori_img = batch['ori_images']
    names = batch['names']
    points_gt = batch['skel_points']
    PMs = batch['PM_label']
    heatmap_pred = F.sigmoid(batch_out['heatmap_pred'])
    sorted_points_pred = batch_out['points_pred'].cpu().numpy()
    vertices_score = batch_out['vertices_score'].cpu().numpy()

    PM_pred = batch_out['PM_pred']
    PM_pred = F.sigmoid(PM_pred)
    PM_pred[PM_pred > thr] = 1
    PM_pred[PM_pred <= thr] = 0

    BN = min(10, heatmap.shape[0])
    # BN = ori_img.shape[0]
    for n in range(0, BN):
        heatmap_gt = heatmap.squeeze(1).detach().cpu().numpy()[n, ...]
        heatmap_predict = heatmap_pred.squeeze(1).detach().cpu().numpy()[n, ...]
        if np.sum(heatmap_gt) == 0:
            continue

        points_true = points_gt[n]
        points_pred = sorted_points_pred[n]
        points_score = vertices_score[n]

        points_pred = np.asarray(points_pred)
        # points_true_show = np.asarray([coord for seq in points_true for coord in seq])

        PM_true = PMs.squeeze(1).cpu().detach().numpy()[n, ...]
        adj_index_pred = PM_pred.cpu().detach().numpy()[n, ...]


        image = ori_img.numpy()[n, ...]

        img = image.astype(np.uint8)
        img_point_pred = img.copy()
        img_point_ture = img.copy()
        img_PM_pred = img.copy()
        img_PM_true = img.copy()
        pair_idx_pred = np.argwhere(adj_index_pred == 1)
        pair_idx_pred = [p for p in pair_idx_pred if p[0] != p[1]]
        for pair in pair_idx_pred:
            if points_score[pair[0]] > score and points_score[pair[1]] > score:
                # pt = points_pred[pair[0]]
                # adj_pt = points_pred[pair[1]]
                # cv2.circle(img_PM_pred, (int(pt[1]), int(pt[0])), 4, (255, 0, 0), -1)
                # cv2.circle(img_PM_pred, (int(adj_pt[1]), int(adj_pt[0])), 4, (255, 0, 0), -1)
                # cv2.arrowedLine(img_PM_pred, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])),
                #                 (242, 203, 5), thickness=2, tipLength=0.3)
                pt = points_pred[pair[0]]
                adj_pt = points_pred[pair[1]]
                mid = (pt + adj_pt) / 2
                cv2.line(img_PM_pred, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])), (5, 203, 242),2)
                cv2.arrowedLine(img_PM_pred, (int(pt[1]), int(pt[0])), (int(mid[1]), int(mid[0])),
                                (5, 203, 242), thickness=2, tipLength=0.3)
                cv2.circle(img_PM_pred, (int(pt[1]), int(pt[0])), 4, (242, 203, 5), -1)
                cv2.circle(img_PM_pred, (int(adj_pt[1]), int(adj_pt[0])), 4, (242, 203, 5), -1)

            else:
                continue

        for cnt in points_true:
            for i in range(len(cnt) - 1):
                pt = cnt[i]
                next = cnt[i + 1]
                cv2.circle(img_point_ture, (int(pt[1]), int(pt[0])), 4, (255, 0, 0), -1)
                cv2.circle(img_point_ture, (int(next[1]), int(next[0])), 4, (255, 0, 0), -1)
                cv2.arrowedLine(img_point_ture, (int(pt[1]), int(pt[0])), (int(next[1]), int(next[0])), (242, 203, 5),
                                thickness=2, tipLength=0.3)


        pair_idx = np.argwhere(PM_true == 1)
        pair_idx = [p for p in pair_idx if p[0] != p[1]]
        for pair in pair_idx:
            pt = points_pred[pair[0]]
            adj_pt = points_pred[pair[1]]
            cv2.circle(img_PM_true, (int(pt[1]), int(pt[0])), 4, (255, 0, 0), -1)
            cv2.circle(img_PM_true, (int(adj_pt[1]), int(adj_pt[0])), 4, (255, 0, 0), -1)
            cv2.arrowedLine(img_PM_true, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])),
                            (242, 203, 5), thickness=2, tipLength=0.3)
            # cv2.line(img_PM_true, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])),
            #                 (255, 0, 0),
            #                 thickness=1)

        norm_img = np.zeros(heatmap_predict.shape)
        heatmap_predict = heatmap_predict.astype(np.float32)
        norm_img = cv2.normalize(heatmap_predict, norm_img, 0, 255, cv2.NORM_MINMAX)
        norm_img = np.asarray(norm_img, dtype=np.uint8)

        norm_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
        # norm_img = cv2.cvtColor(norm_img,cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image[:, :, [2, 1, 0]])
        img_add = cv2.addWeighted(image, 0.3, norm_img, 0.7, 0)
        # heat_save_path = os.path.join(heat_dir, '{}.png'.format(names[n]))
        # cv2.imwrite(heat_save_path, img_add)

        plt.figure()
        plt.subplot(231)
        plt.title('points pred')
        plt.imshow(img_point_pred)
        plt.axis('off')
        plt.plot(points_pred[:, 1], points_pred[:, 0], 'o', markersize=1, color='r')
        plt.subplot(232)
        plt.title('points gt')
        plt.imshow(img_point_ture)
        plt.axis('off')
        plt.subplot(234)
        plt.title('CW connection pred')
        plt.imshow(img_PM_pred)
        plt.axis('off')
        plt.subplot(235)
        plt.title('CW connection gt')
        plt.imshow(img_PM_true)
        plt.axis('off')
        plt.subplot(233)
        plt.title('Heatmap gt')
        plt.imshow(heatmap_gt)
        plt.axis('off')
        plt.subplot(236)
        plt.title('Heatmap pred')
        plt.imshow(img_add)
        plt.axis('off')
        img_save_path = os.path.join(vis_dir, 'E{}_{}.png'.format(epoch, names[n]))
        plt.savefig(img_save_path, dpi=400, bbox_inches='tight')
        # plt.show()
        del points_true, points_pred, img_point_pred, img_point_ture, img_PM_true, img_PM_pred

def visualize_road_eval(batch, batch_out, vis_dir=None, mode='train',
                                 threshold=0.45, score=0.6):
    vis_dir = os.path.join(vis_dir, mode)
    os.makedirs(vis_dir, exist_ok=True)
    show_dir = os.path.join(vis_dir, 'show')
    os.makedirs(show_dir, exist_ok=True)
    heat_dir = os.path.join(vis_dir, 'heat')
    os.makedirs(heat_dir, exist_ok=True)
    infer_dir = os.path.join(vis_dir, 'infer')
    os.makedirs(infer_dir, exist_ok=True)

    thr = threshold
    heatmap = batch['heatmaps']
    ori_img = batch['ori_images']
    names = batch['names']
    points_gt = batch['skel_points']
    PMs = batch['PM_label']
    heatmap_pred = F.sigmoid(batch_out['heatmap_pred'])
    sorted_points_pred = batch_out['points_pred'].cpu().numpy()
    vertices_score = batch_out['vertices_score'].cpu().numpy()

    PM_pred = batch_out['PM_pred']
    PM_pred = F.sigmoid(PM_pred)
    PM_pred[PM_pred > thr] = 1
    PM_pred[PM_pred <= thr] = 0

    # BN = min(10, heatmap.shape[0])
    BN = ori_img.shape[0]
    for n in range(0, BN):
        heatmap_gt = heatmap.squeeze(1).detach().cpu().numpy()[n, ...]
        heatmap_predict = heatmap_pred.squeeze(1).detach().cpu().numpy()[n, ...]

        image = ori_img.numpy()[n, ...]
        img = image.astype(np.uint8)
        
        norm_img = np.zeros(heatmap_predict.shape)
        heatmap_predict = heatmap_predict.astype(np.float32)
        norm_img = cv2.normalize(heatmap_predict, norm_img, 0, 255, cv2.NORM_MINMAX)
        norm_img = np.asarray(norm_img, dtype=np.uint8)

        norm_img = cv2.applyColorMap(norm_img,cv2.COLORMAP_JET)
        # norm_img = cv2.cvtColor(norm_img,cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image[:,:,[2,1,0]])
        img_add = cv2.addWeighted(image,0.3,norm_img,0.7,0)
        
        heat_save_path = os.path.join(heat_dir, '{}.png'.format(names[n]))
        cv2.imwrite(heat_save_path, img_add)

        if np.sum(heatmap_gt) == 0:
            continue

        points_true = points_gt[n]
        points_pred = sorted_points_pred[n]
        points_score = vertices_score[n]

        points_pred = np.asarray(points_pred)
        # points_true_show = np.asarray([coord for seq in points_true for coord in seq])

        PM_true = PMs.squeeze(1).cpu().detach().numpy()[n, ...]
        adj_index_pred = PM_pred.cpu().detach().numpy()[n, ...]


        image = ori_img.numpy()[n, ...]

        img = image.astype(np.uint8)
        img_point_pred = img.copy()
        img_point_ture = img.copy()
        img_PM_pred = img.copy()
        img_PM_pred = cv2.cvtColor(img_PM_pred,cv2.COLOR_RGB2BGR)
        img_PM_true = img.copy()
        pair_idx_pred = np.argwhere(adj_index_pred == 1)
        pair_idx_pred = [p for p in pair_idx_pred if p[0] != p[1]]
        for pair in pair_idx_pred:
            if points_score[pair[0]] > score and points_score[pair[1]] > score:
                pt = points_pred[pair[0]]
                adj_pt = points_pred[pair[1]]
                mid = (pt + adj_pt) / 2
                # cv2.line(img_PM_pred, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])), (5, 203, 242),2)
                # cv2.arrowedLine(img_PM_pred, (int(pt[1]), int(pt[0])), (int(mid[1]), int(mid[0])),
                #                 (5, 203, 242), thickness=2, tipLength=0.3)
                # cv2.circle(img_PM_pred, (int(pt[1]), int(pt[0])), 4, (242, 203, 5), -1)
                # cv2.circle(img_PM_pred, (int(adj_pt[1]), int(adj_pt[0])), 4, (242, 203, 5), -1)

                cv2.line(img_PM_pred, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])), (242, 203, 5),2)
                cv2.arrowedLine(img_PM_pred, (int(pt[1]), int(pt[0])), (int(mid[1]), int(mid[0])),
                                (242, 203, 5), thickness=2, tipLength=0.3)
                cv2.circle(img_PM_pred, (int(pt[1]), int(pt[0])), 4, (5, 203, 242), -1)
                cv2.circle(img_PM_pred, (int(adj_pt[1]), int(adj_pt[0])), 4, (5, 203, 242), -1)

            else:
                continue

        for cnt in points_true:
            for i in range(len(cnt) - 1):
                pt = cnt[i]
                next = cnt[i + 1]
                cv2.circle(img_point_ture, (int(pt[1]), int(pt[0])), 4, (255, 0, 0), -1)
                cv2.circle(img_point_ture, (int(next[1]), int(next[0])), 4, (255, 0, 0), -1)
                cv2.arrowedLine(img_point_ture, (int(pt[1]), int(pt[0])), (int(next[1]), int(next[0])), (242, 203, 5),
                                thickness=2, tipLength=0.3)


        pair_idx = np.argwhere(PM_true == 1)
        pair_idx = [p for p in pair_idx if p[0] != p[1]]
        for pair in pair_idx:
            pt = points_pred[pair[0]]
            adj_pt = points_pred[pair[1]]
            cv2.circle(img_PM_true, (int(pt[1]), int(pt[0])), 4, (255, 0, 0), -1)
            cv2.circle(img_PM_true, (int(adj_pt[1]), int(adj_pt[0])), 4, (255, 0, 0), -1)
            cv2.arrowedLine(img_PM_true, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])),
                            (242, 203, 5), thickness=2, tipLength=0.3)
            # cv2.line(img_PM_true, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])),
            #                 (255, 0, 0),
            #                 thickness=1)

        
        infer_save_path = os.path.join(infer_dir, '{}.png'.format(names[n]))
        cv2.imwrite(infer_save_path, img_PM_pred)

        plt.figure()
        plt.subplot(231)
        plt.title('points pred')
        plt.imshow(img_point_pred)
        plt.axis('off')
        plt.plot(points_pred[:, 1], points_pred[:, 0], 'o', markersize=1, color='r')
        plt.subplot(232)
        plt.title('points gt')
        plt.imshow(img_point_ture)
        plt.axis('off')
        plt.subplot(234)
        plt.title('CW connection pred')
        img_PM_pred = cv2.cvtColor(img_PM_pred,cv2.COLOR_RGB2BGR)
        plt.imshow(img_PM_pred)
        plt.axis('off')
        plt.subplot(235)
        plt.title('CW connection gt')
        plt.imshow(img_PM_true)
        plt.axis('off')
        plt.subplot(233)
        plt.title('Heatmap gt')
        plt.imshow(heatmap_gt)
        plt.axis('off')
        plt.subplot(236)
        plt.title('Heatmap pred')
        plt.imshow(img_add)
        plt.axis('off')
        img_save_path = os.path.join(show_dir, '{}.png'.format(names[n]))
        plt.savefig(img_save_path, dpi=400, bbox_inches='tight')
        # plt.show()
        del points_true, points_pred, img_point_pred, img_point_ture, img_PM_true, img_PM_pred

def PM2poly(batch, batch_out, epoch=None, vis=False, vis_dir=None, mode='train', dilate_pixels=5,retrun_polys=False):
    vis_dir = os.path.join(vis_dir, mode)
    os.makedirs(vis_dir, exist_ok=True)

    ori_img = batch['ori_images']
    names = batch['names']
    mask_gt = batch['masks'].squeeze(1).cpu().numpy()
    boundary_mask_gt = batch['boundary_masks'].squeeze(1).cpu().numpy()
    sorted_points_pred = batch_out['points_pred'].cpu().numpy()

    PM_pred = torch.exp(batch_out['PM_pred'])
    PM_pred = scores_to_permutations(PM_pred)


    BN, h, w, _ = ori_img.size()
    mask_pred = np.zeros((BN, h, w), dtype=np.uint8)
    bounadry_mask_pred = np.zeros((BN, h, w), dtype=np.uint8)
    ploys_pred = []
    for n in range(0, BN):
        points_pred = sorted_points_pred[n]
        points_pred = np.asarray(points_pred)
        mask_pred_tmp = mask_pred[n]
        bounadry_mask_pred_tmp = bounadry_mask_pred[n]
        mask_true_tmp = mask_gt[n]
        bounadry_mask_true_tmp = boundary_mask_gt[n]

        if np.sum(mask_true_tmp) == 0:
            vis = False

        adj_index_pred = PM_pred.cpu().detach().numpy()[n, ...]
        DiGraph = AdjMatrix2DiGraph(adj_index_pred, points_pred)
        circles = nx.simple_cycles(DiGraph)
        polys = []
        for list in circles:
            if len(list) > 1:
                poly = [points_pred[i] for i in list]
                polys.append(poly)

        ploys_pred.append(polys)

        img = ori_img.cpu().numpy()[n, ...]
        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)


        for i in range(len(polys)):
            tmp = polygon2mask((img.shape[0], img.shape[1]), polys[i])
            mask_pred_tmp += tmp
        mask_pred_tmp = np.uint8(mask_pred_tmp)
        mask_pred_tmp[mask_pred_tmp > 0] = 1

        for cnt in polys:
            for p in range(len(cnt)):
                if p == len(cnt) - 1:
                    cv2.line(bounadry_mask_pred_tmp, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[0][1]), int(cnt[0][0])),
                             (255, 0, 0), 1)
                else:
                    cv2.line(bounadry_mask_pred_tmp, (int(cnt[p][1]), int(cnt[p][0])),
                             (int(cnt[p + 1][1]), int(cnt[p + 1][0])),
                             (255, 0, 0), 1)

        bounadry_mask_pred_tmp = cv2.GaussianBlur(np.float32(bounadry_mask_pred_tmp), ksize=(dilate_pixels, dilate_pixels), sigmaX=1, sigmaY=1)
        bounadry_mask_pred_tmp[bounadry_mask_pred_tmp > 0] = 1
        mask_pred[n] = mask_pred_tmp
        bounadry_mask_pred[n] = bounadry_mask_pred_tmp

        # img_save_path = os.path.join(vis_dir, '{}.tif'.format(names[n]))
        # cv2.imwrite(img_save_path, mask_pred_tmp*255)

        if vis:
            plt.figure()
            plt.subplot(231)
            plt.imshow(img)
            plt.axis('off')
            plt.subplot(232)
            plt.title('mask_pred')
            plt.imshow(mask_pred_tmp * 255)
            plt.axis('off')
            plt.subplot(233)
            plt.title('mask_true')
            plt.imshow(mask_true_tmp * 255)
            plt.axis('off')
            plt.subplot(234)
            plt.title('bounadry_mask_pred')
            plt.imshow(bounadry_mask_pred_tmp * 255)
            plt.axis('off')
            plt.subplot(235)
            plt.title('bounadry_mask_true')
            plt.imshow(bounadry_mask_true_tmp * 255)
            plt.axis('off')
            img_save_path = os.path.join(vis_dir, 'E{}_{}.png'.format(epoch, names[n]))
            plt.savefig(img_save_path, dpi=400, bbox_inches='tight')
            # plt.show()

    if retrun_polys:
        return torch.from_numpy(mask_pred), torch.from_numpy(bounadry_mask_pred), ploys_pred
    else:
        return torch.from_numpy(mask_pred), torch.from_numpy(bounadry_mask_pred)


def PM2road(batch, batch_out, epoch=None, vis=False, vis_dir=None, mode='train', threshold=0.45,
            score=0.6,dilate_pixels=5):
    vis_dir = os.path.join(vis_dir, mode)
    os.makedirs(vis_dir, exist_ok=True)

    ori_img = batch['ori_images']
    names = batch['names']
    mask_gt = batch['masks'].squeeze(1).cpu().numpy()
    boundary_mask_gt = batch['boundary_masks'].squeeze(1).cpu().numpy()
    sorted_points_pred = batch_out['points_pred'].cpu().numpy()
    vertices_score = batch_out['vertices_score'].cpu().numpy()
    thr = threshold
    PM_pred = F.sigmoid(batch_out['PM_pred'])
    PM_pred[PM_pred >= thr] = 1
    PM_pred[PM_pred < thr] = 0

    BN, h, w, _ = ori_img.size()
    mask_pred = np.zeros((BN, h, w), dtype=np.uint8)
    bounadry_mask_pred = np.zeros((BN, h, w), dtype=np.uint8)
    for n in range(0, BN):
        points_pred = sorted_points_pred[n]
        points_pred = np.asarray(points_pred)
        points_score = vertices_score[n]

        adj_index_pred = PM_pred.cpu().detach().numpy()[n, ...]


        img = ori_img.cpu().numpy()[n, ...]
        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)

        bounadry_mask_pred_tmp = bounadry_mask_pred[n]
        mask_true_tmp = mask_gt[n]
        bounadry_mask_true_tmp = boundary_mask_gt[n]

        pair_idx = np.argwhere(adj_index_pred == 1)
        pair_idx = [p for p in pair_idx if p[0] != p[1]]
        for pair in pair_idx:
            if points_score[pair[0]] > score and points_score[pair[1]] > score:
                pt = points_pred[pair[0]]
                adj_pt = points_pred[pair[1]]
                cv2.line(bounadry_mask_pred_tmp, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])),
                         (255, 0, 0), 1)
            else:
                continue


        bounadry_mask_pred_tmp = cv2.GaussianBlur(np.float32(bounadry_mask_pred_tmp), ksize=(dilate_pixels, dilate_pixels), sigmaX=1, sigmaY=1)
        bounadry_mask_pred_tmp[bounadry_mask_pred_tmp > 0] = 1
        mask_pred_tmp = bounadry_mask_pred_tmp
        mask_pred[n] = mask_pred_tmp
        bounadry_mask_pred[n] = bounadry_mask_pred_tmp

        # img_save_path = os.path.join(vis_dir, '{}.tif'.format(names[n]))
        # cv2.imwrite(img_save_path, mask_pred_tmp*255)
        
        if vis:
            plt.figure()
            plt.subplot(231)
            plt.imshow(img)
            plt.axis('off')
            plt.subplot(232)
            plt.title('mask_pred')
            plt.imshow(mask_pred_tmp * 255)
            plt.axis('off')
            plt.subplot(233)
            plt.title('mask_true')
            plt.imshow(mask_true_tmp * 255)
            plt.axis('off')
            plt.subplot(234)
            plt.title('bounadry_mask_pred')
            plt.imshow(bounadry_mask_pred_tmp * 255)
            plt.axis('off')
            plt.subplot(235)
            plt.title('bounadry_mask_true')
            plt.imshow(bounadry_mask_true_tmp * 255)
            plt.axis('off')
            img_save_path = os.path.join(vis_dir, 'E{}_{}.png'.format(epoch, names[n]))
            plt.savefig(img_save_path, dpi=400, bbox_inches='tight')
            # plt.show()

    return torch.from_numpy(mask_pred), torch.from_numpy(bounadry_mask_pred)

def output_road(batch, batch_out,  vis=False, vis_dir=None, threshold=0.45,score=0.6,line_thickness=3):
    vis_root = os.path.join(vis_dir, 'skel')
    # vis_root = os.path.join(vis_root,'no_weighting')
    img_save_dir = os.path.join(vis_root,'img')
    true_skel_dir = os.path.join(vis_root,'gt')
    pred_skel_dir = os.path.join(vis_root,'pred')
    os.makedirs(vis_root, exist_ok=True)
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(true_skel_dir, exist_ok=True)
    os.makedirs(pred_skel_dir, exist_ok=True)

    # pred_skel_dir = '/root/autodl-tmp/Datasets/Massachusetts/cropped300/valid/seg_preds/TopDiG'
    # os.makedirs(pred_skel_dir, exist_ok=True)

    ori_img = batch['ori_images']
    names = batch['names']
    points_gt = batch['skel_points']
    sorted_points_pred = batch_out['points_pred'].cpu().numpy()
    vertices_score = batch_out['vertices_score'].cpu().numpy()
    thr = threshold
    # PM_pred = torch.exp(batch_out['PM_pred'])
    PM_pred = F.sigmoid(batch_out['PM_pred'])
    PM_pred[PM_pred >= thr] = 1
    PM_pred[PM_pred < thr] = 0

    BN, h, w, _ = ori_img.size()
    skel_mask_true = np.zeros((BN,h,w),dtype=np.uint8)
    skel_mask_pred = np.zeros((BN, h, w), dtype=np.uint8)
    for n in range(0, BN):
        points_true = points_gt[n]
        points_pred = sorted_points_pred[n]
        points_pred = np.asarray(points_pred)
        points_score = vertices_score[n]

        adj_index_pred = PM_pred.cpu().detach().numpy()[n, ...]

        img = ori_img.cpu().numpy()[n, ...]
        img = img.astype(np.uint8)[:, :, [2, 1, 0]]
        img = np.ascontiguousarray(img)

        true_skel_mask_tmp = skel_mask_true[n]
        pred_skel_mask_tmp = skel_mask_pred[n]

        pair_idx = np.argwhere(adj_index_pred == 1)
        pair_idx = [p for p in pair_idx if p[0] != p[1]]
        for pair in pair_idx:
            if points_score[pair[0]] > score and points_score[pair[1]] > score:
                pt = points_pred[pair[0]]
                adj_pt = points_pred[pair[1]]
                cv2.line(pred_skel_mask_tmp, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])),
                         (255, 0, 0), line_thickness)
            else:
                continue

        for cnt in points_true:
            for i in range(len(cnt) - 1):
                pt = cnt[i]
                next = cnt[i + 1]
                cv2.line(true_skel_mask_tmp, (int(pt[1]), int(pt[0])), (int(next[1]), int(next[0])), (255, 0, 0),line_thickness)
        skel_mask_pred[n] = pred_skel_mask_tmp
        skel_mask_true[n] = true_skel_mask_tmp

        if vis:
            # plt.subplot(131)
            # plt.imshow(img)
            # plt.subplot(132)
            # plt.imshow(true_skel_mask_tmp)
            # plt.subplot(133)
            # plt.imshow(pred_skel_mask_tmp)
            # plt.show()
            # cv2.imwrite(os.path.join(img_save_dir,'{}.jpg'.format(names[n])),img)
            # cv2.imwrite(os.path.join(true_skel_dir,'{}.jpg'.format(names[n])),true_skel_mask_tmp)
            cv2.imwrite(os.path.join(pred_skel_dir, '{}.jpg'.format(names[n])), pred_skel_mask_tmp)


def output_poly(batch, batch_out,  vis=False, vis_dir=None):
    vis_root = os.path.join(vis_dir, 'skel')
    # vis_root = os.path.join(vis_root,'no_weighting')
    img_save_dir = os.path.join(vis_root,'img')
    true_skel_dir = os.path.join(vis_root,'gt')
    pred_skel_dir = os.path.join(vis_root,'pred')
    pred_seg_dir = os.path.join(vis_root,'seg')
    os.makedirs(img_save_dir,exist_ok=True)
    os.makedirs(true_skel_dir,exist_ok=True)
    os.makedirs(pred_skel_dir,exist_ok=True)
    os.makedirs(pred_seg_dir,exist_ok=True)


    ori_img = batch['ori_images']
    names = batch['names']
    skel_mask_true = batch['boundary_masks']

    mask_pred = batch_out['mask_pred'].squeeze().cpu().numpy()
    boundary_mask_pred = batch_out['boundary_mask_pred'].squeeze().cpu().numpy()
    PM_pred = torch.exp(batch_out['PM_pred'])
    PM_pred = scores_to_permutations(PM_pred)

    BN, h, w, _ = ori_img.size()
    for n in range(0, BN):

        img = ori_img.cpu().numpy()[n, ...]
        img = img.astype(np.uint8)[:, :, [2, 1, 0]]
        img = np.ascontiguousarray(img)

        true_skel_mask_tmp = skel_mask_true[n]
        pred_mask_tmp = mask_pred[n]
        pred_skel_mask_tmp = boundary_mask_pred[n]


        if vis:
            # plt.subplot(131)
            # plt.imshow(img)
            # plt.subplot(132)
            # plt.imshow(true_skel_mask_tmp)
            # plt.subplot(133)
            # plt.imshow(pred_skel_mask_tmp)
            # plt.show()
            # cv2.imwrite(os.path.join(img_save_dir,'{}.jpg'.format(names[n])),img)
            # cv2.imwrite(os.path.join(true_skel_dir,'{}.jpg'.format(names[n])),true_skel_mask_tmp)
            cv2.imwrite(os.path.join(pred_seg_dir, '{}.tif'.format(names[n])), pred_mask_tmp)
            # cv2.imwrite(os.path.join(pred_skel_dir, '{}.jpg'.format(names[n])), pred_skel_mask_tmp)


def visualize_infer_polygons(basename, image, outputs, vis_dir=None, mode='test'):
    vis_dir = os.path.join(vis_dir, mode)
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    points_pred = np.squeeze(outputs['points_pred'].cpu().numpy())
    PM_pred = torch.exp(outputs['PM_pred'])
    PM_pred = scores_to_permutations(PM_pred)

    adj_index_pred = np.squeeze(PM_pred.cpu().detach().numpy())

    img = image.astype(np.uint8)
    img_PM_pred = img.copy()

    DiGraph_pred = AdjMatrix2DiGraph(adj_index_pred, points_pred)
    circles_pred = nx.simple_cycles(DiGraph_pred)
    polys_pred = []
    for list in circles_pred:
        if len(list) > 1:
            poly_tmp_pred = [points_pred[i] for i in list]
            polys_pred.append(poly_tmp_pred)

    for cnt in polys_pred:
        for p in range(len(cnt)):
            if p == len(cnt) - 1:
                mid = (cnt[p] + cnt[0]) / 2
                cv2.line(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[0][1]), int(cnt[0][0])),
                                (5, 203, 242),2)
                cv2.arrowedLine(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(mid[1]), int(mid[0])),
                                (5, 203, 242), thickness=2, tipLength=0.3)
                cv2.circle(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), 4, (242, 203, 5), -1)
                cv2.circle(img_PM_pred, (int(cnt[0][1]), int(cnt[0][0])), 4, (242, 203, 5), -1)

            else:
                mid = (cnt[p] + cnt[p + 1]) / 2
                cv2.line(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[p + 1][1]), int(cnt[p + 1][0])),
                                (5, 203, 242), thickness=2)
                cv2.arrowedLine(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), (int(mid[1]), int(mid[0])),
                                (5, 203, 242), thickness=2, tipLength=0.3)
                cv2.circle(img_PM_pred, (int(cnt[p][1]), int(cnt[p][0])), 4, (242, 203, 5), -1)
                cv2.circle(img_PM_pred, (int(cnt[p + 1][1]), int(cnt[p + 1][0])), 4, (242, 203, 5), -1)


    img_PM_pred = img_PM_pred[:, :, [2,1,0]]
    img_save_path = os.path.join(vis_dir, '{}.png'.format(basename))
    cv2.imwrite(img_save_path,img_PM_pred)
    del points_pred, img_PM_pred


def visualize_infer_roads(basename, image, outputs, vis_dir=None, mode='test', thr=0.6,score=0.6):
    vis_dir = os.path.join(vis_dir, mode)
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    points_pred = np.squeeze(outputs['points_pred'].cpu().numpy())
    points_score = np.squeeze(outputs['vertices_score'].cpu().numpy())
    PM_pred = outputs['PM_pred']
    PM_pred = F.sigmoid(PM_pred)

    adj_index_pred = np.squeeze(PM_pred.cpu().detach().numpy())

    adj_index_pred[adj_index_pred > thr] = 1
    adj_index_pred[adj_index_pred <= thr] = 0

    img = image.astype(np.uint8)
    img_PM_pred = img.copy()

    pair_idx_pred = np.argwhere(adj_index_pred == 1)
    pair_idx_pred = [p for p in pair_idx_pred if p[0] != p[1]]
    for pair in pair_idx_pred:
        if points_score[pair[0]] > score and points_score[pair[1]] > score:
            pt = points_pred[pair[0]]
            adj_pt = points_pred[pair[1]]
            mid = (pt + adj_pt) / 2
            cv2.line(img_PM_pred, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])),
                     (5, 203, 242), thickness=2)
            cv2.arrowedLine(img_PM_pred, (int(pt[1]), int(pt[0])), (int(mid[1]), int(mid[0])),
                            (5, 203, 242), thickness=2, tipLength=0.3)
            cv2.circle(img_PM_pred, (int(pt[1]), int(pt[0])), 4, (242, 203, 5), -1)
            cv2.circle(img_PM_pred, (int(adj_pt[1]), int(adj_pt[0])), 4, (242, 203, 5), -1)
        else:
            continue

    img_PM_pred = img_PM_pred[:, :, [2, 1, 0]]
    img_save_path = os.path.join(vis_dir, '{}.png'.format(basename))
    cv2.imwrite(img_save_path,img_PM_pred)
    del points_pred,img_PM_pred


def visualize_Adjacency_Matrix(S, PM_pred,PM_true,args,epoch, batch):
    S_show = F.sigmoid(S.clone()).cpu().detach().numpy()
    P_show = torch.exp(PM_pred.clone()).cpu().detach().numpy()
    P_true = PM_true.squeeze(1).cpu().detach().numpy()
    
    BN = min(5, S_show.shape[0])
    for i in range(BN):

        # plt.figure()
        # plt.matshow(P_show[0] * 255, cmap=plt.cm.gray)
        # plt.savefig('./CW.png', dpi=400, bbox_inches='tight')
        # plt.show()
        # plt.figure()
        # plt.matshow(P_show[0].transpose() * 255, cmap=plt.cm.gray)
        # plt.savefig('./CCW.png', dpi=400, bbox_inches='tight')
        # plt.show()

        plt.figure()
        plt.subplot(221)
        plt.title('score_map')
        plt.imshow(S_show[i])
        plt.subplot(222)
        plt.title('PM_pred')
        plt.imshow(P_show[i])
        plt.subplot(223)
        plt.title('CW_PM_true')
        plt.imshow(P_true[i] * 255)
        plt.subplot(224)
        plt.title('CCW_PM_true')
        plt.imshow(P_true[i].transpose() * 255)
        save_root = os.path.join(args.vis_root, 'PM_val')
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        save_path = os.path.join(save_root, 'E{}_{}.png'.format(epoch, batch['names'][i]))
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        # plt.show()


def visualize_feature_map(batch, batch_feature_map, vis_dir=None, mode='test'):
    vis_dir = vis_dir + '/' + mode + '/' + 'feature_map'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    b, c, _, _ = batch_feature_map.size()
    names = batch['names']
    for i in range(b):
        feature_map = batch_feature_map[i]
        feature_map = feature_map.detach().cpu().numpy()
        feature_map_num = feature_map.shape[0]
        row_num = int(np.ceil(np.sqrt(feature_map_num)))
        plt.figure()
        for index in range(1, feature_map_num + 1):
            plt.subplot(row_num, row_num, index)
            plt.imshow(feature_map[index - 1])
            plt.axis('off')
        img_save_path = os.path.join(vis_dir, '{}.png'.format(names[i]))
        plt.savefig(img_save_path, bbox_inches='tight')
        # plt.show()


def visualize_keypoints(batch, batch_out, epoch, vis_dir=None, mode='train', N=256, score=0.8, env=None):
    vis_dir = vis_dir + '/' + mode + '/' + 'nodes'
    os.makedirs(vis_dir, exist_ok=True)

    heatmap = batch['heatmaps']
    ori_img = batch['ori_images']
    names = batch['names']
    heatmap_pred = batch_out['heatmap_pred']
    # ske_pred = F.sigmoid(ske_pred)
    # heatmap_pred = F.sigmoid(heatmap_pred)
    # median = torch.median(heatmap_pred)
    # heatmap_pred[heatmap_pred < median] = 0

    BN = min(10, heatmap.shape[0])
    # BN = heatmap.shape[0]
    for n in range(0, BN):
        heatmap_predict = heatmap_pred.squeeze(1).cpu().detach().numpy()[n, ...]
        heatmap_gt = heatmap.squeeze(1).cpu().numpy()[n, ...]
        if np.sum(heatmap_gt) == 0:
            continue
        # points_pred = getPoints(heatmap_predict)
        points_pred = gpu_NMS(heatmap_predict, N=N,gap=env.gap)
        # points_pred = peak_finder(heatmap_predict)
        points_true = np.argwhere(heatmap_gt == heatmap_gt.max())
        if len(points_true) == 0 or len(points_pred[0]) == 0:
            continue
        points_pred = np.asarray(points_pred)
        points_true = np.asarray(points_true)

        image = ori_img.numpy()[n, ...]
        img = image.astype(np.uint8)
        img_show = img.copy()
        img_show_pred = img.copy()

        points_pred = (points_pred / heatmap_predict.shape[-1]) * image.shape[0]
        points_true = (points_true / heatmap_predict.shape[-1]) * image.shape[0]
        # for pt in points_pred:
        #     cv2.circle(img_show_pred, (int(pt[1]), int(pt[0])), 2, (255, 0, 0), -1)
        # plt.imshow(img_show_pred)
        # plt.show()
        img_save_path = os.path.join(vis_dir, '{}.png'.format(names[n]))
        plt.figure()
        plt.subplot(221)
        plt.imshow(img_show)
        plt.title('Points_pred')
        plt.plot(points_pred[:, 1], points_pred[:, 0], 'o', markersize=1, color='r')
        plt.axis('off')
        plt.subplot(222)
        plt.imshow(img_show_pred)
        plt.title('Points_GT')
        plt.plot(points_true[:, 1], points_true[:, 0], 'o', markersize=1, color='b')
        plt.axis('off')
        plt.subplot(223)
        plt.title('Heatmap_pred')
        plt.imshow(heatmap_predict)
        plt.axis('off')
        plt.subplot(224)
        plt.title('Heatmap_GT')
        plt.imshow(heatmap_gt)
        plt.axis('off')
        plt.savefig(img_save_path, dpi=400, bbox_inches='tight')
        # plt.show()


def visualize_sideouts(batch, batch_out, epoch, vis_dir=None, mode='test'):
    vis_dir = vis_dir + '/' + mode + '/' + 'edge_sideouts'
    # vis_dir = vis_dir + '/' + mode + '/' + 'heat'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    names = batch['names']
    ori_img = batch['ori_images']
    edge_gt = batch['boundary_masks']
    heatmap_gt = batch['heatmaps']
    outs = batch_out['outputs']
    BN = min(10, ori_img.shape[0])
    # BN = heatmap.shape[0]
    cls = ['building', 'woodland', 'water']
    for n in range(0, BN):
        outs_tmp = [o[n, ...] for o in outs]
        outs_tmp = [np.squeeze(o.cpu().detach().numpy()) for o in outs_tmp]
        edge_gt_ = np.squeeze(edge_gt.cpu().detach().numpy()[n])
        heatmap_gt_ = heatmap_gt.cpu().detach().numpy()[n]
        image = ori_img.numpy()[n, ...]
        img = image.astype(np.uint8)
        side1, side2, side3, side5, final = outs_tmp[0], outs_tmp[1], outs_tmp[2], outs_tmp[3], outs_tmp[4]
        # side1 = save_heatmap(side1, img)
        # side2 = save_heatmap(side2, img)
        # side3 = save_heatmap(side3, img)
        # cv2.imwrite(os.path.join(vis_dir, '{}_side1.png'.format(names[n])), side1)
        # cv2.imwrite(os.path.join(vis_dir, '{}_side2.png'.format(names[n])), side2)
        # cv2.imwrite(os.path.join(vis_dir, '{}_side3.png'.format(names[n])), side3)
        # # cv2.imwrite(os.path.join(vis_dir,'{}_side1.png'.format(names[n])), np.uint8(side1*255))
        # # cv2.imwrite(os.path.join(vis_dir, '{}_side2.png'.format(names[n])), np.uint8(side2*255))
        # # cv2.imwrite(os.path.join(vis_dir, '{}_side3.png'.format(names[n])), np.uint8(side3*255))
        # cv2.imwrite(os.path.join(vis_dir, '{}_edge_gt.png'.format(names[n])), np.uint8(edge_gt_*255))
        # for i in range(len(cls)):
        #     side5_cls = side5[i]
        #     final_cls = final[i]
        #     side5_cls = save_heatmap(side5_cls, img)
        #     final_cls = save_heatmap(final_cls, img)
        #     cv2.imwrite(os.path.join(vis_dir, '{}_side5_{}.png'.format(names[n], cls[i])), side5_cls)
        #     cv2.imwrite(os.path.join(vis_dir, '{}_final_{}.png'.format(names[n], cls[i])), final_cls)
        #     # cv2.imwrite(os.path.join(vis_dir, '{}_side5_{}.png'.format(names[n], cls[i])), np.uint8(side5[i]*255))
        #     # cv2.imwrite(os.path.join(vis_dir, '{}_final_{}.png'.format(names[n], cls[i])), np.uint8(final[i]*255))
        #     cv2.imwrite(os.path.join(vis_dir, '{}_heatmap_gt_{}.png'.format(names[n], cls[i])), np.uint8(heatmap_gt_[i]*255))

        plt.figure()
        plt.subplot(231)
        plt.imshow(img)
        plt.title('image')
        plt.axis('off')
        plt.subplot(232)
        plt.imshow(side1)
        plt.title('side1')
        plt.axis('off')
        plt.subplot(233)
        plt.imshow(side2)
        plt.title('side2')
        plt.axis('off')
        plt.subplot(234)
        plt.imshow(side3)
        plt.title('side3')
        plt.axis('off')
        plt.subplot(235)
        plt.imshow(side5)
        plt.title('side5')
        plt.axis('off')
        plt.subplot(236)
        plt.imshow(final)
        plt.title('final')
        plt.axis('off')
        img_save_path = os.path.join(vis_dir, '{}.png'.format(names[n]))
        plt.savefig(img_save_path, bbox_inches='tight')