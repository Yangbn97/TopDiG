import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from tqdm import tqdm
from utils.NMS import gpu_NMS
from utils.data_utils import remove_junctions, remove_extra_points
from itertools import permutations, product
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def scores_to_permutations(scores):
    """
    Input a batched array of scores and returns the hungarian optimized
    permutation matrices.
    """
    B, N, N = scores.shape

    scores = scores.detach().cpu().numpy()
    perm = np.zeros_like(scores)
    for b in range(B):
        r, c = linear_sum_assignment(-scores[b])
        perm[b, r, c] = 1
    return torch.tensor(perm)


def AdjMatrix2DiGraph(AdjMatrix, points):
    G = nx.DiGraph()
    N = AdjMatrix.shape[0]
    for i in range(0, N):
        G.add_node(i, pos=(points[i]))
    edges = np.argwhere(AdjMatrix == 1)
    for edge in edges:
        G.add_edge(edge[0], edge[1])

    return G


def polys2AdjMatrix(polys):
    lens = [len(ct) for ct in polys]
    N = sum(lens)
    polys_np = np.concatenate(polys, axis=0)
    AdjMatrix = np.zeros((N, N), dtype=np.float32)
    start_idx = 0
    for i in range(len(polys)):
        cnt = polys[i]
        if lens[i] + start_idx > N:
            contours = polys[:i]
            break
        for j in range(len(cnt)):
            if j < len(cnt) - 1:
                AdjMatrix[j + start_idx][j + 1 + start_idx] = 1
            else:
                AdjMatrix[j + start_idx][0 + start_idx] = 1

        start_idx += lens[i]

    return AdjMatrix, polys_np


def Build_PermutationMatrix_poly(contours, N=256):
    lens = [len(ct) for ct in contours]
    n = sum(lens)
    CW_PM = np.zeros((N, N), dtype=np.float32)
    for i in range(len(contours)):
        cnt = contours[i]
        for j in range(len(cnt)):
            if j < len(cnt) - 1:
                CW_PM[cnt[j]][cnt[j + 1]] = 1
            else:
                CW_PM[cnt[j]][cnt[0]] = 1

    for row in range(N):
        if np.any(CW_PM[row]):
            continue
        else:
            CW_PM[row][row] = 1

    return CW_PM


def Build_PermutationMatrix_road(mls, key_record, N=256):
    lens = [len(ct) for ct in mls]
    n = sum(lens)

    PM = np.zeros((N, N), dtype=np.float32)
    junctions = mls[0]
    for i in range(1, len(mls)):
        line = mls[i]

        if len(line) == 0 and len(junctions):  # only start and end
            start_node_idx = int(key_record[i][0])
            end_node_idx = int(key_record[i][1])
            if 0 < junctions[start_node_idx] and 0 < junctions[end_node_idx] \
                    and start_node_idx > 0 and end_node_idx > 0:
                PM[junctions[start_node_idx]][junctions[end_node_idx]] = 1
            continue

        if key_record[i][0] >= 0 and len(junctions):  # if the start of this line is a junction.
            juc_node_idx = int(key_record[i][0])
            if junctions[juc_node_idx] > 0:
                PM[junctions[juc_node_idx]][line[0]] = 1
        for j in range(len(line)):
            if j < len(line) - 1:
                PM[line[j]][line[j + 1]] = 1
            else:
                if key_record[i][1] >= 0 and len(junctions):  # if the end of this line is a junction
                    juc_node_idx = int(key_record[i][1])
                    if junctions[juc_node_idx] > 0:
                        PM[line[-1]][junctions[juc_node_idx]] = 1

    for row in range(N):
        if np.any(PM[row]):
            continue
        else:
            PM[row][row] = 1

    return PM


def is_polygon_clockwise(polygon):
    rolled_polygon = np.roll(polygon, shift=1, axis=0)
    double_signed_area = np.sum((rolled_polygon[:, 0] - polygon[:, 0]) * (rolled_polygon[:, 1] + polygon[:, 1]))
    if 0 < double_signed_area:
        return True
    else:
        return False


def orient_polygon(polygon, orientation="CW"):
    poly_is_orientated_cw = is_polygon_clockwise(polygon)
    if (poly_is_orientated_cw and orientation == "CCW") or (not poly_is_orientated_cw and orientation == "CW"):
        return np.flip(polygon, axis=0)
    else:
        return polygon


def orient_polygons(polygons, orientation="CW"):
    return [orient_polygon(polygon, orientation=orientation) for polygon in polygons]


def getPoints(heatmap, N=256, score=0.8, get_score=False, gap=10):
    assert N or score
    BN = heatmap.shape[0]
    points_pred = torch.zeros((BN, N, 2)).to(heatmap.device)
    points_score = torch.zeros((BN, N, 1)).to(heatmap.device)
    heatmap = F.sigmoid(heatmap)
    for n in range(0, BN):
        heatmap_predict = heatmap.clone().squeeze(1).cpu().detach().numpy()[n, ...]
        # points = gpu_NMS(heatmap_predict, N=N)

        points = gpu_NMS(heatmap_predict, N=N, gap=gap)
        points = np.asarray(points, dtype=np.float32)
        points_pred[n] = torch.from_numpy(points)
        if get_score:
            score_tmp = [heatmap_predict[int(pt[0])][int(pt[1])] for pt in points]
            score_tmp = np.asarray(score_tmp, dtype=np.float32)
            points_score[n] = torch.from_numpy(score_tmp[:N, None])
            # sorted_id = sorted(range(len(score_tmp)), key=lambda k: score_tmp[k], reverse=True)
            # points = np.stack([points[i] for i in sorted_id])
            # score_tmp = np.stack([score_tmp[i] for i in sorted_id])
            # plt.imshow(heatmap_predict)
            # for i in range(len(score_tmp)):
            #     coor = points[i]
            #     sc = round(score_tmp[i], 4)
            #     plt.text(int(coor[1]), int(coor[0]), str(sc), fontsize=5, color='r')
            # plt.savefig('./records/water_local/score_test.png',dpi=500)
            # plt.show()
    if get_score:
        return points_pred, points_score
    else:
        return points_pred


def getVerticesDescriptors_grid(vertices_pred, feature_map):
    vertices_pred = vertices_pred.type_as(feature_map)
    row, col = feature_map.shape[2], feature_map.shape[3]
    vertices_xy = torch.stack([torch.flip(k, [1]) for k in vertices_pred.clone()], dim=0)  # convert (h,w) to (x,y)
    vertices_xy[..., 0] = vertices_xy[..., 0] / (col / 2.) - 1  # scale coordinates to (-1,1)
    vertices_xy[..., 1] = vertices_xy[..., 1] / (row / 2.) - 1
    feature_vector = torch.nn.functional.grid_sample(feature_map, vertices_xy[:, None, :, :], mode='bilinear',
                                                     align_corners=True)
    feature_vector = feature_vector.squeeze(2).permute(0, 2, 1)  # (bs,N,feature_dim)
    feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=2)
    return feature_vector


def getDescriptors_point(vertices_pred, feature_map):
    B, N, _ = vertices_pred.shape
    sel_desc = 0
    for b in range(B):
        b_desc = feature_map[b]
        b_graph = vertices_pred[b].long()

        # Extract descriptors
        b_desc = b_desc[:, b_graph[:, 0], b_graph[:, 1]]

        # Concatenate descriptors in batches
        if b == 0:
            sel_desc = b_desc.unsqueeze(0)
        else:
            sel_desc = torch.cat((sel_desc, b_desc.unsqueeze(0)), dim=0)

    return sel_desc.permute(0, 2, 1)


def getAdjMatrix_poly(points_pred_orig, vertices_score, polys, score=0.6, tol=10):
    if torch.is_tensor(points_pred_orig):
        points_pred_orig = points_pred_orig.cpu().numpy()
    if torch.is_tensor(vertices_score):
        vertices_score = vertices_score.cpu().numpy()
    BN = len(points_pred_orig)
    N = len(points_pred_orig[0])
    PM_sorted = np.zeros((BN, N, N))
    for i in range(BN):
        points_pred = points_pred_orig[i]
        pred_tree = cKDTree(points_pred)
        point_score = vertices_score[i]
        img_polys = polys[i]
        pred_polys = []
        visited_list = []
        for cnt in img_polys:
            pred_cnt = []
            for j in range(len(cnt)):
                dis, idx = pred_tree.query(cnt[j], k=3)
                # dis += tol
                idx = [idx[n] for n in range(len(dis)) if dis[n] <= tol]
                if not len(idx):
                    continue
                score_tmp = point_score[idx]
                score_ = max(score_tmp)
                idx = idx[score_tmp.tolist().index(score_)]
                if idx not in visited_list:
                    visited_list.append(idx)
                    pred_cnt.append(idx)
            pred_polys.append(pred_cnt)

        PM_tmp = Build_PermutationMatrix_poly(pred_polys, N=N)
        PM_sorted[i] = PM_tmp

    return PM_sorted


def get_polygons(outputs):
    points_pred = np.squeeze(outputs['points_pred'].cpu().numpy())
    PM_pred = torch.exp(outputs['PM_pred'])
    PM_pred = scores_to_permutations(PM_pred)

    adj_index_pred = np.squeeze(PM_pred.cpu().detach().numpy())

    DiGraph_pred = AdjMatrix2DiGraph(adj_index_pred, points_pred)
    circles_pred = nx.simple_cycles(DiGraph_pred)
    polys_pred = []
    for list in circles_pred:
        if len(list) > 1:
            poly_tmp_pred = [points_pred[i] for i in list]
            if len(poly_tmp_pred) > 2:
                if not np.array_equal(poly_tmp_pred[0], poly_tmp_pred[-1]):
                    poly_tmp_pred.append(poly_tmp_pred[0])
                polys_pred.append(poly_tmp_pred)

    return polys_pred


def get_lines(outputs, thr=0.6, score=0.6):
    points_pred = np.squeeze(outputs['points_pred'].cpu().numpy())
    points_score = np.squeeze(outputs['points_score'].cpu().numpy())
    PM_pred = F.sigmoid(outputs['PM_pred'])
    PM_pred[PM_pred >= thr] = 1
    PM_pred[PM_pred < thr] = 0
    adj_index_pred = np.squeeze(PM_pred.cpu().detach().numpy())
    row, col = np.diag_indices_from(adj_index_pred)
    adj_index_pred[row, col] = 0
    pair_idx = np.argwhere(adj_index_pred == 1)
    pair_idx = [p for p in pair_idx if p[0] != p[1]]
    # DiGraph = AdjMatrix2DiGraph(adj_index_pred, points_pred)
    DiGraph = nx.DiGraph()
    for pair in pair_idx:
        # pt = points_pred[pair[0]]
        # adj_pt = points_pred[pair[1]]
        if points_score[pair[0]] > score and points_score[pair[1]] > score:
            DiGraph.add_node(pair[0], pos=(points_pred[pair[0]]))
            DiGraph.add_node(pair[1], pos=(points_pred[pair[1]]))
            DiGraph.add_edge(pair[0], pair[1])
    relabeled_graph = nx.relabel.convert_node_labels_to_integers(DiGraph, first_label=0, ordering='default')
    attrs = nx.get_node_attributes(relabeled_graph, 'pos')
    nodes = []
    if len(attrs) > 0:
        nodes = [attrs[i] for i in range(len(attrs))]
    return relabeled_graph, nodes


def rm_patch_margin_nodes(patch_instances, patch_size=512):
    new_patch_instances = []
    for instance in patch_instances:
        ins_np = np.stack(instance, axis=0)
        rows = ins_np[:, 0]
        cols = ins_np[:, 1]
        rows_flag = (rows > 0) * (rows < patch_size - 3)
        cols_flag = (cols > 0) * (cols < patch_size - 3)
        idx = rows_flag * cols_flag
        ins_selected = ins_np[idx]
        if len(ins_selected) > 0:
            new_patch_instances.append(ins_selected)
    return new_patch_instances


def getAdjMatrix_road(points_pred_orig, vertices_score, points_gt_orig, junction_nodes, tol=10):
    if torch.is_tensor(points_pred_orig):
        points_pred_orig = points_pred_orig.cpu().numpy()
    if torch.is_tensor(vertices_score):
        vertices_score = vertices_score.cpu().numpy()
    BN = len(points_pred_orig)
    N = len(points_pred_orig[0])
    PM_sorted = np.zeros((BN, N, N))
    tol = 10
    for i in range(BN):
        points_pred = points_pred_orig[i]
        pred_tree = cKDTree(points_pred)
        point_score = vertices_score[i]
        points_gt = points_gt_orig[i]
        junctions = junction_nodes[i]

        points_gt_no_juc, juc_record = remove_junctions(points_gt, junctions)

        pred_roads = []
        visited_list = []
        for v in range(len(points_gt_no_juc)):
            cnt = points_gt_no_juc[v]
            pred_road = []
            for j in range(len(cnt)):
                dis, idx = pred_tree.query(cnt[j], k=3)
                idx = [idx[n] for n in range(len(dis)) if dis[n] <= tol]
                if not len(idx):
                    if v == 0:
                        pred_road.append(-1)
                    continue
                score_tmp = point_score[idx]
                score_ = max(score_tmp)
                idx = idx[score_tmp.tolist().index(score_)]
                if idx not in visited_list:
                    visited_list.append(idx)
                    pred_road.append(idx)
                else:
                    if v == 0:
                        pred_road.append(-1)
            pred_roads.append(pred_road)

        pred_roads = remove_extra_points(pred_roads, N)

        PM_tmp = Build_PermutationMatrix_road(pred_roads, key_record=juc_record, N=N)
        PM_sorted[i] = PM_tmp

    return PM_sorted


def ReplicationPad(x1, x2, h, w):
    h = int(h)
    w = int(w)
    top1 = np.abs(x1.size()[2] - h)
    left1 = np.abs(x1.size()[3] - w)
    top2 = np.abs(x2.size()[2] - h)
    left2 = np.abs(x2.size()[3] - w)
    # slice_top1=h-top1
    # slice_left1=w-left1
    # slice_top2 = h-top2
    # slice_left2 = w-left2
    ReplicationPad1 = nn.ReplicationPad2d(padding=(left1, 0, top1, 0))
    ReplicationPad2 = nn.ReplicationPad2d(padding=(left2, 0, top2, 0))
    if h > x2.size()[2] or w > x2.size()[3]: x2 = ReplicationPad2(x2)
    if h < x2.size()[2] or w < x2.size()[3]: x2 = x2[:, :, :h, :w]
    if x1.size()[2] < h or x1.size()[3] < w: x1 = ReplicationPad1(x1)
    if h < x1.size()[2] or w < x1.size()[3]: x1 = x1[:, :, :h, :w]
    return x1, x2


def region2boundary(mask):
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded[1:mask.shape[0] + 1, 1:mask.shape[1] + 1] = mask
    dist = cv2.distanceTransform(src=padded, distanceType=cv2.DIST_L2, maskSize=5)
    dist[dist != 1] = 0
    dist[dist == 1] = 255
    boundary_mask = dist[1:mask.shape[0] + 1, 1:mask.shape[1] + 1]
    return boundary_mask
    # dist = cv2.distanceTransform(src=mask, distanceType=cv2.DIST_L2, maskSize=5)
    # dist[dist != 1] = 0
    # dist[dist == 1] = 255
    # return dist


def correct_points(dict, tor=10):
    curr_nodes = dict['instances']
    pre_nodes = dict['nbr_instances']['instances_np']
    if len(pre_nodes) == 0:
        return dict
    pre_nodes_tree = [cKDTree(ls) for ls in pre_nodes]
    for idx in range(len(pre_nodes_tree)):
        tree = pre_nodes_tree[idx]
        previous_nodes = pre_nodes[idx]
        for i in range(len(curr_nodes)):
            line = curr_nodes[i]
            for j in range(len(line)):
                pt = line[j]
                dis, ii = tree.query(pt)
                if dis < tor:
                    curr_nodes[i][j] = previous_nodes[ii]

    curr_nodes_np = np.concatenate([np.stack(ins, axis=0) for ins in curr_nodes], axis=0)
    dict['instances'] = curr_nodes
    dict['instances_np'] = curr_nodes_np
    return dict


def update_graph_nodes_pos(dict):
    for i in range(len(dict)):
        patch_ins = dict[i]
        nodes = patch_ins['instances_np']
        DiGraph = patch_ins['DiGraph']
        # graph_nodes = attrs = nx.get_node_attributes(DiGraph, 'pos')
        for j in range(len(nodes)):
            DiGraph.nodes[j]['pos'] = nodes[j]

        dict[i]['DiGraph'] = DiGraph
    return dict


def coord_region2full(patch_points, patch_info):
    start = [patch_info[0], patch_info[1]]
    points_new = []
    for instance in patch_points:
        instance_new = []
        for pt in instance:
            pt_ = np.asarray([pt[0] + start[0], pt[1] + start[1]])
            instance_new.append(pt_)
        points_new.append(instance_new)

    return points_new


def joinPatchNodes(patch_dict, row_patch_count, col_patch_count):
    dicts = []
    # FullGraph = nx.DiGraph()
    par_disable = True if len(patch_dict.values()) == 1 else False
    for instance_dict in tqdm(patch_dict.values(), desc="Joining patch nodes", leave=True, disable=par_disable):
        row_ids = instance_dict['region'][4]
        col_ids = instance_dict['region'][5]
        patch_polys = instance_dict['instances']

        if len(patch_polys) == 0:
            continue

        nbr_instance = []
        nbr_dict = {'instances_np': [],
                    'region': []}
        instance_dict['nbr_instances'] = nbr_dict
        if row_ids == col_ids == 0:  # left corner, no neighbour patches
            dicts.append(instance_dict)
            continue
        elif row_ids == 0 and col_ids != 0:  # first row, 1 neighbour patches
            neighbours = [row_ids, col_ids - 1]
            nbr_instance.append(patch_dict[f'{neighbours[0]}_{neighbours[1]}'])
        elif row_ids != 0 and col_ids == 0:  # first column without left top, 2 neighbour patches
            neighbours = [[row_ids - 1, col_ids], [row_ids - 1, col_ids + 1]]
            # ids = row_patch_count * (row_ids - 1)
            for id in neighbours:
                nbr_instance.append(patch_dict[f'{id[0]}_{id[1]}'])
        # elif row_ids == row_patch_count - 1 and col_ids > 0: #last row without left bottom
        #     neighbours = [[row_ids, col_ids-1],[row_ids-1, col_ids-1], [row_ids-1, col_ids],[row_ids-1, col_ids+1]]
        #     # ids = [n[0]*row_patch_count + n[1] for n in neighbours]
        #     for id in neighbours:
        #         nbr_instance.append(polys_dict[f'{id[0]}_{id[1]}'])
        elif 0 < row_ids and col_ids == col_patch_count - 1:  # last column without right bottom, 3 neighbour patches
            neighbours = [[row_ids - 1, col_ids], [row_ids - 1, col_ids - 1], [row_ids, col_ids - 1]]
            # ids = [n[0] * row_patch_count + n[1] for n in neighbours]
            for id in neighbours:
                nbr_instance.append(patch_dict[f'{id[0]}_{id[1]}'])
        elif 0 < row_ids and 0 < col_ids < col_patch_count - 1:  # common patches and last row without left bottom, 4 neighbour patches
            neighbours = [[row_ids, col_ids - 1], [row_ids - 1, col_ids - 1], [row_ids - 1, col_ids],
                          [row_ids - 1, col_ids + 1]]
            # ids = [n[0] * row_patch_count + n[1] for n in neighbours]
            for id in neighbours:
                nbr_instance.append(patch_dict[f'{id[0]}_{id[1]}'])
        else:
            print('ERROR!')

        for item in nbr_instance:
            nbr_poly = item['instances_np']
            if len(nbr_poly) == 0:
                continue
            nbr_dict['instances_np'].append(item['instances_np'])
            nbr_dict['region'].append(item['region'])

        instance_dict['nbr_instances'] = nbr_dict
        instance_dict_corrected = instance_dict
        instance_dict_corrected = correct_points(instance_dict,tor=15)
        dicts.append(instance_dict_corrected)

    return dicts


def compose_DiGraph(all_patch_dicts):
    all_patch_nodes = [ins['instances_np'] for ins in all_patch_dicts]
    all_patch_graphs = [ins['DiGraph'] for ins in all_patch_dicts]

    FullDiGraph = all_patch_graphs[0]
    for i in range(1, len(all_patch_nodes)):
        tmp_graph = all_patch_graphs[i]
        tmp_nodes = all_patch_nodes[i]
        tmp_count = len(tmp_nodes)
        tmp_nodes_index = [c for c in range(tmp_count)]
        tmp_index_change_record = [0 for _ in range(tmp_count)]
        attrs = nx.get_node_attributes(FullDiGraph, 'pos')
        explored_nodes = [attrs[key] for key in attrs.keys()]
        curr_id = len(explored_nodes)
        tree = cKDTree(explored_nodes)
        for j in range(tmp_count):
            dis, ii = tree.query(tmp_nodes[j])
            if dis == 0:
                tmp_nodes_index[j] = ii
                tmp_index_change_record[j] = 1
                tmp_count -= 1
        count = 0
        tmp_nodes_index_new = []
        for k in range(len(tmp_nodes_index)):
            if tmp_index_change_record[k] == 0:
                tmp_nodes_index_new.append(curr_id + count)
                count += 1
            else:
                tmp_nodes_index_new.append(tmp_nodes_index[k])
        # tmp_nodes_index_new = [tmp_nodes_index[k] + curr_id if not tmp_index_change_record[k] else tmp_nodes_index[k] for k in range(len(tmp_nodes_index))]
        mapping = {}
        for p in range(len(tmp_nodes_index_new)):
            mapping[p] = tmp_nodes_index_new[p]
        tmp_graph_relabeled = nx.relabel_nodes(tmp_graph, mapping, copy=True)
        FullDiGraph = nx.compose(FullDiGraph, tmp_graph_relabeled)

    return FullDiGraph
