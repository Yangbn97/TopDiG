import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from utils.NMS import gpu_NMS
from itertools import permutations, product
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


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
    if get_score:
        return points_pred, points_score
    else:
        return points_pred


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



def region2boundary(mask):
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded[1:mask.shape[0] + 1, 1:mask.shape[1] + 1] = mask
    dist = cv2.distanceTransform(src=padded, distanceType=cv2.DIST_L2, maskSize=5)
    dist[dist != 1] = 0
    dist[dist == 1] = 255
    boundary_mask = dist[1:mask.shape[0] + 1, 1:mask.shape[1] + 1]
    return boundary_mask


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
    for instance_dict in patch_dict.values():
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
