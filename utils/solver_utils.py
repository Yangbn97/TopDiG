import torch
import numpy as np
import cv2
import time
from skimage.draw import polygon2mask
import math
from shapely.geometry import Point, LineString,MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils.poly_utils import *
from utils.visualize import *
from utils.data_utils import normalization

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


def poly2mask(im, poly, mode_test=False):
    '''
    :param im:
    :param poly:
    :return:
    '''
    b, c, m, n = im.shape
    label = np.zeros((m, n), dtype=np.uint8)
    for i in range(len(poly)):
        tmp = polygon2mask((m, n), poly[i])
        label += tmp
    if mode_test:
        return label
    else:
        return torch.from_numpy(label).unsqueeze(1).float()


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

            patch_boundingbox = np.array([row_slice_begin, col_slice_begin, row_slice_end, col_slice_end, i, j], dtype=np.int16)
            assert row_slice_end - row_slice_begin == col_slice_end - col_slice_begin == patch_res, "ERROR: patch does not have the requested shape"
            patch_boundingboxes.append(patch_boundingbox)

    return patch_boundingboxes,row_patch_count,col_patch_count

def predict_heatmap_WithOverlap_simple(args, img, feature_dim=64, patch_size=512, overlap_rate=1/4):
    '''

    :param model: a trained model
    :param img: a path for an image
    :param patch_size:
    :param overlap_rate:
    :return:
    '''
    # subsidiary value for the prediction of an image with overlap
    boder_value = int(patch_size * overlap_rate / 2)
    double_bv = boder_value * 2
    stride_value = patch_size - double_bv
    most_value = stride_value + boder_value

    # an image for prediction
    m, n, _ = img.shape
    tmp = (m - double_bv) // stride_value  # 剔除重叠部分相当于无缝裁剪
    new_m = tmp if (m - double_bv) % stride_value == 0 else tmp + 1
    tmp = (n - double_bv) // stride_value
    new_n = tmp if (n - double_bv) % stride_value == 0 else tmp + 1
    FullPredict = np.zeros((m, n), dtype=np.uint8)
    FullFeaturemap = np.zeros((feature_dim, m, n), dtype=np.float32)
    FullHeatmap = np.zeros((m, n), dtype=np.float32)
    for i in range(new_m):
        for j in range(new_n):
            if i == new_m - 1 and j != new_n - 1:
                tmp_img = img[
                          -patch_size:,
                          j * stride_value:((j + 1) * stride_value + double_bv), :]
            elif i != new_m - 1 and j == new_n - 1:
                tmp_img = img[
                          i * stride_value:((i + 1) * stride_value + double_bv),
                          -patch_size:, :]
            elif i == new_m - 1 and j == new_n - 1:
                tmp_img = img[
                          -patch_size:,
                          -patch_size:, :]
            else:
                tmp_img = img[
                          i * stride_value:((i + 1) * stride_value + double_bv),
                          j * stride_value:((j + 1) * stride_value + double_bv), :]
            tmp_img = np.array(tmp_img, np.float32).transpose(2, 0, 1) / 255.0
            tmp_img = torch.from_numpy(tmp_img).unsqueeze(0)
            tmp_img = tmp_img.cuda()

            with torch.no_grad():
                # tmp_img = tmp_img.cuda().unsqueeze(0)
                with autocast():
                    heatmap_pred, feature_map = args.detection_model(tmp_img)

            if i == 0 and j == 0:  # 左上角
                FullHeatmap[0:most_value, 0:most_value] = heatmap_pred[0:most_value, 0:most_value]
                FullFeaturemap[:, 0:most_value, 0:most_value] = feature_map[: 0:most_value, 0:most_value]

            elif i == 0 and j == new_n-1:  # 右上角
                FullHeatmap[0:most_value, -most_value:] = heatmap_pred[0:most_value, boder_value:]
                FullFeaturemap[:, 0:most_value, -most_value:] = feature_map[:, 0:most_value, boder_value:]
            elif i == 0 and j != 0 and j != new_n - 1:  # 第一行
                FullHeatmap[0:most_value, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                    heatmap_pred[0:most_value, boder_value:most_value]
                FullFeaturemap[:, 0:most_value, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                    feature_map[0:most_value, boder_value:most_value]

            elif i == new_m - 1 and j == 0:  # 左下角
                FullHeatmap[-most_value:, 0:most_value] = heatmap_pred[boder_value:, :-boder_value]
                FullFeaturemap[-most_value:, 0:most_value] = feature_map[boder_value:, :-boder_value]
            elif i == new_m - 1 and j == new_n - 1:  # 右下角
                FullHeatmap[-most_value:, -most_value:] = heatmap_pred[boder_value:, boder_value:]
                FullFeaturemap[-most_value:, -most_value:] = feature_map[boder_value:, boder_value:]
            elif i == new_m - 1 and j != 0 and j != new_n - 1:  # 最后一行
                FullHeatmap[-most_value:, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                    heatmap_pred[boder_value:, boder_value:-boder_value]
                FullFeaturemap[-most_value:, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                    feature_map[boder_value:, boder_value:-boder_value]

            elif j == 0 and i != 0 and i != new_m - 1:  # 第一列
                FullHeatmap[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, 0:most_value] = \
                    heatmap_pred[boder_value:-boder_value, 0:-boder_value]
                FullFeaturemap[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, 0:most_value] = \
                    feature_map[boder_value:-boder_value, 0:-boder_value]
            elif j == new_n - 1 and i != 0 and i != new_m - 1:  # 最后一列
                FullHeatmap[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, -most_value:] = \
                    heatmap_pred[boder_value:-boder_value, boder_value:]
                FullFeaturemap[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, -most_value:] = \
                    feature_map[boder_value:-boder_value, boder_value:]
            else:  # 中间情况
                FullHeatmap[
                boder_value + i * stride_value:boder_value + (i + 1) * stride_value,
                boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                    heatmap_pred[boder_value:-boder_value, boder_value:-boder_value]
                FullFeaturemap[
                boder_value + i * stride_value:boder_value + (i + 1) * stride_value,
                boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                    feature_map[boder_value:-boder_value, boder_value:-boder_value]
    return FullHeatmap,FullFeaturemap


def predict_heatmap_WithOverlap_weight(args, img, feature_dim=64, patch_size=512, overlap_rate=1/4):
    '''

    :param model: a trained model
    :param img: a path for an image
    :param patch_size:
    :param overlap_rate:
    :return:
    '''
    # subsidiary value for the prediction of an image with overlap
    boder_value = int(patch_size * overlap_rate / 2)
    double_bv = boder_value * 2
    stride_value = patch_size - double_bv

    # an image for prediction
    height, width, _ = img.shape
    FullFeaturemap = torch.zeros((feature_dim,height, width), dtype=torch.float32).cuda()
    FullHeatmap = torch.zeros((height, width), dtype=torch.float32).cuda()
    weight_map = torch.zeros((height, width), dtype=torch.float32).cuda()

    patch_boundingboxes, row_patch_count, col_patch_count = compute_patch_boundingboxes((height, width),stride=stride_value,patch_res=patch_size)
    patch_weights = np.ones((patch_size + 2, patch_size + 2),dtype=np.float32)

    patch_weights[0, :] = 0
    patch_weights[-1, :] = 0
    patch_weights[:, 0] = 0
    patch_weights[:, -1] = 0
    patch_weights = scipy.ndimage.distance_transform_edt(patch_weights)
    patch_weights = patch_weights[1:-1, 1:-1]
    patch_weights = torch.from_numpy(patch_weights).cuda()

    # Predict on each patch and save in outputs:
    for bbox in tqdm(patch_boundingboxes, desc="Extracting nodes on patches", leave=True):
        # Crop data
        tmp_img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        tmp_img = np.array(tmp_img, np.float32).transpose(2, 0, 1) / 255.0
        tmp_img = torch.from_numpy(tmp_img).unsqueeze(0)
        tmp_img = tmp_img.cuda()
        # Detect nodes
        with torch.no_grad():
            with autocast():
                tmp_heatmap, tmp_feature = args.detection_model(tmp_img)

        patch_weights = patch_weights.to(tmp_heatmap.device)
        FullHeatmap[bbox[0]:bbox[2], bbox[1]:bbox[3]] += patch_weights * tmp_heatmap.squeeze()
        FullFeaturemap[:,bbox[0]:bbox[2], bbox[1]:bbox[3]] += patch_weights * tmp_feature.squeeze()
        weight_map[bbox[0]:bbox[2], bbox[1]:bbox[3]] += patch_weights

    FullHeatmap /= weight_map
    FullFeaturemap /= weight_map

    return FullHeatmap,FullFeaturemap






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
    par_disable = True if len(patch_boundingboxes) == 1 else False
    for bbox in tqdm(patch_boundingboxes, desc="Running on patches", leave=True, disable=par_disable):
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
                if isinstance(tmp_heatmap, list):
                    tmp_heatmap = tmp_heatmap[-1]
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
    par_disable = True if len(patch_boundingboxes) == 1 else False
    for bbox in tqdm(patch_boundingboxes, desc="Running on patches", leave=True, disable=par_disable):
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
                if isinstance(tmp_heatmap, list):
                    tmp_heatmap = tmp_heatmap[-1]
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





