import os
import time

import networkx as nx
import numpy as np
import cv2
from skimage.draw import polygon2mask
import math
from shapely.geometry import Point, LineString,MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils.poly_utils import *

def draw_poly(mask, poly):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask


def get_axis_patch_count(length, stride, patch_res):
    total_double_padding = patch_res - stride
    patch_count = max(1, int(math.ceil((length - total_double_padding) / stride)))
    return patch_count

def compute_patch_boundingboxes(image_size, stride, patch_res):
    """

    @param image_size:
    @param stride:
    @param patch_res:
    @return: [[row_start, col_start, row_end, col_end], ...]
    """
    im_rows = image_size[0]
    im_cols = image_size[1]

    row_patch_count = get_axis_patch_count(im_rows, stride, patch_res)
    col_patch_count = get_axis_patch_count(im_cols, stride, patch_res)

    patch_boundingboxes = []
    for i in range(0, row_patch_count):
        if i < row_patch_count - 1:
            row_slice_begin = i * stride
            row_slice_end = row_slice_begin + patch_res
        else:
            row_slice_end = im_rows
            row_slice_begin = row_slice_end - patch_res
        for j in range(0, col_patch_count):
            if j < col_patch_count - 1:
                col_slice_begin = j*stride
                col_slice_end = col_slice_begin + patch_res
            else:
                col_slice_end = im_cols
                col_slice_begin = col_slice_end - patch_res

            patch_boundingbox = np.array([row_slice_begin, col_slice_begin, row_slice_end, col_slice_end, i, j], dtype=np.int)
            assert row_slice_end - row_slice_begin == col_slice_end - col_slice_begin == patch_res, "ERROR: patch does not have the requested shape"
            patch_boundingboxes.append(patch_boundingbox)

    return patch_boundingboxes,row_patch_count,col_patch_count


def predict_poly_graph_WithOverlap_simple(args, img, patch_size=512, intersect_area=100):
    img_BGR = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    # subsidiary value for the prediction of an image with overlap
    boder_value = int(intersect_area / 2)
    double_bv = boder_value * 2
    stride_value = patch_size - double_bv
    most_value = stride_value + boder_value

    # an image for prediction
    height, width, _ = img.shape

    polys_dict = {}
    patch_boundingboxes, row_patch_count, col_patch_count = compute_patch_boundingboxes((height, width),
                                                                                        stride=stride_value,
                                                                                        patch_res=patch_size)
    model_time = []
    for bbox in tqdm(patch_boundingboxes, desc="Running on patches", leave=True):
        # Crop data
        tmp_img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        ori_img = tmp_img.copy()
        tmp_img = np.array(tmp_img, np.float32).transpose(2, 0, 1) / 255.0
        tmp_img = torch.from_numpy(tmp_img).unsqueeze(0)
        tmp_img = tmp_img.cuda()
        # Detect nodes
        model_start_time = time.time()
        with torch.no_grad():
            with autocast():
                tmp_heatmap, tmp_feature = args.detection_model(tmp_img)
                vertices, vertices_score = getPoints(tmp_heatmap, args.configs['Model']['NUM_POINTS'], get_score=True,
                                     gap=args.gap)
                S, PM_pred = args.match_model(tmp_img, tmp_feature, vertices)

        model_end_time = time.time()
        model_time.append(model_end_time - model_start_time)

        tmp_polys = get_polygons({'points_pred': vertices, 'points_score': vertices_score, 'PM_pred': PM_pred})
        tmp_polys_selected = tmp_polys

        if len(tmp_polys_selected) > 0:
            tmp_polys_selected = coord_region2full(tmp_polys_selected, bbox)
            tmp_polys_np = np.concatenate([np.stack(poly, axis=0) for poly in tmp_polys_selected], axis=0)
        else:
            tmp_polys_np = []
        instance = {'instances': tmp_polys_selected,
                    'instances_np': tmp_polys_np,
                    'region': bbox}
        # polys_dict.append(instance)
        polys_dict[f'{bbox[4]}_{bbox[5]}'] = instance

    all_patch_dicts = joinPatchNodes(polys_dict, row_patch_count, col_patch_count)
    polygons = [Polygon(cnt) for ins in all_patch_dicts for cnt in ins['instances']]
    polygons = [geom if geom.is_valid else geom.buffer(0) for geom in polygons]
    merged_polygons = unary_union(polygons) if len(polygons) > 1 else polygons

    if isinstance(merged_polygons, list):
        merged_polygons = merged_polygons[0] if len(merged_polygons) else merged_polygons

    return merged_polygons, np.sum(model_time)

def predict_line_graph_WithOverlap_simple(args, img, patch_size=512, intersect_area=100):
    img_BGR = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    # subsidiary value for the prediction of an image with overlap
    boder_value = int(intersect_area / 2)
    double_bv = boder_value * 2
    stride_value = patch_size - double_bv
    most_value = stride_value + boder_value

    # an image for prediction
    height, width, _ = img.shape

    patch_dict = {}
    patch_boundingboxes, row_patch_count, col_patch_count = compute_patch_boundingboxes((height, width),
                                                                                        stride=stride_value,
                                                                                        patch_res=patch_size)
    model_time = []
    for bbox in tqdm(patch_boundingboxes, desc="Running on patches", leave=True):
        # Crop data
        tmp_img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        ori_img = tmp_img.copy()
        tmp_img = np.array(tmp_img, np.float32).transpose(2, 0, 1) / 255.0
        tmp_img = torch.from_numpy(tmp_img).unsqueeze(0)
        tmp_img = tmp_img.cuda()
        # Detect nodes
        model_start_time = time.time()
        with torch.no_grad():
            with autocast():
                tmp_heatmap, tmp_feature = args.detection_model(tmp_img)
                vertices, vertices_score = getPoints(tmp_heatmap, args.configs['Model']['NUM_POINTS'], get_score=True,
                                     gap=args.gap)
                S, PM_pred = args.match_model(tmp_img, tmp_feature, vertices)
        model_end_time = time.time()
        model_time.append(model_end_time - model_start_time)

        tmp_line_graphs, patch_nodes = get_lines({'points_pred': vertices,'points_score': vertices_score, 'PM_pred': PM_pred}, thr=args.configs['Model']['threshold'], score=0.6)

        if len(patch_nodes) > 0:
            patch_nodes = coord_region2full([patch_nodes], bbox)
            patch_nodes_np = np.stack(patch_nodes[0], axis=0)
        else:
            patch_nodes_np = []
        instance = {'instances': patch_nodes,
                    'instances_np': patch_nodes_np,
                    'DiGraph': tmp_line_graphs,
                    'region': bbox}
        patch_dict[f'{bbox[4]}_{bbox[5]}'] = instance

    all_patch_dicts = joinPatchNodes(patch_dict, row_patch_count, col_patch_count)
    all_patch_dicts = update_graph_nodes_pos(all_patch_dicts)

    if len(all_patch_dicts):
        FullDiGraph = compose_DiGraph(all_patch_dicts)
        subgraphs = [FullDiGraph.subgraph(c) for c in sorted(nx.weakly_connected_components(FullDiGraph), key=len, reverse=True)]
    else:
        FullDiGraph = nx.DiGraph()
        subgraphs = [FullDiGraph]

    return FullDiGraph, subgraphs, np.sum(model_time)





