import cv2
import cv2 as cv
import numpy as np
import os
import torch
from scipy import interpolate
from skimage.morphology import skeletonize
from skan import skeleton_to_csgraph
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from jsmin import jsmin
from skimage import measure
from scipy.ndimage import convolve
import json
from sknw import build_sknw
import copy


def build_ske(ske, ori_mask):
    graph = build_sknw(ske)
    nodes = graph.nodes()
    pt = [nodes[i]['o'] for i in nodes]
    pointmap = np.zeros((ori_mask.shape[0], ori_mask.shape[1]), np.uint8)
    for i in range(0, len(pt)):
        p = pt[i]
        pointmap[int(p[0])][int(p[1])] = 255

    mls = []
    for (s, e) in graph.edges():
        ps = graph[s][e]['pts']
        for j in range(len(ps)):
            p = ps[j]
            pointmap[int(p[0])][int(p[1])] = 255
        mls.append(ps)

    return mls, pointmap, pt

def sample_line_points_without_N(raw_line, init_stride=15):
    lines = []
    for i in range(len(raw_line)):
        ps = raw_line[i]
        if isinstance(init_stride, list):
            stride = int(init_stride[i])
        else:
            stride = init_stride
        pre = [int(ps[0][0]), int(ps[0][1])]
        end = [int(ps[-1][0]), int(ps[-1][1])]

        n_points = len(ps) // stride
        line = []
        line.append(pre)
        if n_points == 0:
            line.append(end)
            lines.append(line)
            continue
        if n_points == 1:
            mid = [int(ps[len(ps) // 2][0]), int(ps[len(ps) // 2][1])]
            line.append(mid)
            line.append(end)
            lines.append(line)
            continue
        for i in range(1, n_points + 2):
            p = [int(ps[i * stride][0]), int(ps[i * stride][1])]
            # if (i * stride >= len(ps)):
            #     line.append(p)
            #     break

            if (len(ps) - i * stride <= 5):
                break
            if (5 < len(ps) - i * stride <= stride):
                line.append(p)
                break
            line.append(p)
        line.append(end)
        lines.append(line)
    return lines

def sample_skeleton_points(skeletons, N=256, init_stride=15):
    lens = [len(ct) for ct in skeletons]
    points_count = 1000
    stride = init_stride
    lines = []
    while points_count >= N:
        lines = []
        interpolrate_num = 0
        for i in range(len(skeletons)):
            ps = skeletons[i]
            pre = [int(ps[0][0]), int(ps[0][1])]
            end = [int(ps[-1][0]), int(ps[-1][1])]

            n_points = len(ps) // stride
            line = []
            line.append(pre)
            if n_points == 0:
                line.append(end)
                lines.append(line)
                continue
            if n_points == 1:
                mid = [int(ps[len(ps) // 2][0]), int(ps[len(ps) // 2][1])]
                line.append(mid)
                line.append(end)
                lines.append(line)
                interpolrate_num += 1
                continue
            for i in range(1, n_points + 2):
                p = [int(ps[i * stride][0]), int(ps[i * stride][1])]
                if (i * stride >= len(ps)):
                    line.append(p)
                    interpolrate_num += 1
                    break

                if (len(ps) - i * stride <= 5):
                    break
                if (5 < len(ps) - i * stride <= stride):
                    line.append(p)
                    interpolrate_num += 1
                    break
                line.append(p)
            line.append(end)
            lines.append(line)

        points_count = interpolrate_num
        stride += 3

    return lines


def calculate_distance(pt1, pt2):
    dis = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    return dis

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def generate_heatmap(gt, size=5):
    gt = 255 - gt
    D = cv.distanceTransform(gt, cv.DIST_L2, 3)
    A = np.exp(-D * D / (2 * size * size))
    return A


def merge_junction_nodes(mls, junction_nodes, tolerance=5):
    mls_final = copy.deepcopy(mls)
    mls.insert(0, junction_nodes)
    count = 0
    if len(junction_nodes) > 0:
        nodes_tree = cKDTree(junction_nodes)
        for i in range(1, len(mls)):
            line = mls[i]
            start = line[0]
            end = line[-1]
            dis_start, idx_start = nodes_tree.query(start)
            dis_end, idx_end = nodes_tree.query(end)
            if dis_start <= tolerance:
                mls_final[i - 1][0] = junction_nodes[idx_start]
                count += 1
            if dis_end <= tolerance:
                mls_final[i - 1][-1] = junction_nodes[idx_end]
                count += 1
    return mls_final, count

def remove_junctions(roads_true,junction_node,tolerance=5):
    mls = copy.deepcopy(roads_true)
    mls.insert(0, junction_node)
    mls_new = copy.deepcopy(mls)
    key_record = np.zeros((len(mls_new), 2)) - 1
    if len(junction_node) > 0:
        nodes_tree = cKDTree(junction_node)
        for i in range(1, len(mls)):
            line = mls[i]
            start = line[0]
            end = line[-1]
            dis_start, idx_start = nodes_tree.query(start)
            dis_end, idx_end = nodes_tree.query(end)
            if dis_start <= tolerance:
                mls[i][0] = junction_node[idx_start]
                del mls_new[i][0]
                key_record[i][0] = idx_start
            if dis_end <= tolerance:
                mls[i][-1] = junction_node[idx_end]
                del mls_new[i][-1]
                key_record[i][-1] = idx_end

    return mls_new, key_record


def sample_road(img, mask, N=256, init_stride=15, name=None):
    mask[mask != 0] = 1
    mask = np.uint8(mask)


    ske = skeletonize(mask)
    ske = np.uint8(ske)
    ske[ske > 0] = 255

    mls = []
    junction_node = []
    if np.max(ske) != 0 and len(np.argwhere(ske > 0)) > 5:

        # pixel_graph, coordinates, degrees = skeleton_to_csgraph(ske)

        # endpoint_map = np.uint8((degrees == 1) * 255)
        # endpoints = np.argwhere(endpoint_map > 0)
        # node_region = np.uint8((degrees > 2) * 1)

        # junction_node = []
        # label, num = measure.label(node_region, connectivity=2, return_num=True)
        # props = measure.regionprops(label)
        # for prop in props:
        #     centroid = prop.centroid
        #     if len(junction_node):
        #         junction_tree = cKDTree(junction_node)
        #         dis, _ = junction_tree.query(centroid)
        #         if dis > 10:
        #             junction_node.append(np.asarray(centroid, dtype=np.float32))
        #     else:
        #         junction_node.append(np.asarray(centroid, dtype=np.float32))


        # junction_node = np.asarray(junction_node, dtype=np.uint16)

        # Kernel to count 8-connected neighbors
        kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=np.uint8)

        # Convolve to get neighbor counts for each pixel
        neighbor_map = convolve(ske, kernel, mode='constant', cval=0)

        # Keep neighbor counts only for actual skeleton pixels
        neighbor_map = neighbor_map * ske

        # --- 3. Identify Key Point Types ---
        # Endpoints (value=1, neighbors=1)
        endpoints = np.argwhere((ske == 1) & (neighbor_map == 1))

        # Junction points (value=1, neighbors > 2)
        junction_node = np.argwhere((ske == 1) & (neighbor_map > 2))

        img_show = img.copy()
        graph = build_sknw(ske)
        skeletons = [graph[s][e]['pts'] for (s, e) in graph.edges()]

        skeletons, count = merge_junction_nodes(skeletons, junction_node, tolerance=5)

        lens = [len(ct) for ct in skeletons]
        sorted_id = sorted(range(len(lens)), key=lambda k: lens[k], reverse=True)
        skeletons = [skeletons[i] for i in sorted_id]


        sample_num = N - len(junction_node) - len(endpoints)
        if sample_num > 0:
            mls = sample_skeleton_points(skeletons, N=sample_num, init_stride=init_stride)
        else:
            mls = sample_line_points_without_N(skeletons, init_stride=init_stride)

        lens = [len(ct) for ct in mls]
        sorted_id = sorted(range(len(lens)), key=lambda k: lens[k], reverse=True)
        mls = [mls[i] for i in sorted_id]

    return mls, junction_node


def desify_contours(simplfied_contours, dense_contours, stride=20):
    desified_contours = []
    for i in range(len(simplfied_contours)):
        simplfied_cnt = simplfied_contours[i]
        dense_cnt = dense_contours[i]
        desified_cnt = []
        for j in range(len(simplfied_cnt) - 1):
            curr = simplfied_cnt[j]
            next = simplfied_cnt[j + 1]
            if j == 0:
                desified_cnt.append(curr)
            # dis = np.sqrt((curr[0]-next[0])**2+(curr[1]-next[1])**2)
            # curr_ind = np.argwhere(np.all(dense_cnt==curr))
            curr_ind = [n for n, x in enumerate(dense_cnt) if np.all(x == curr)]
            curr_ind = curr_ind[0] if j < len(simplfied_cnt) else curr_ind[-1]
            next_ind = [n for n, x in enumerate(dense_cnt) if np.all(x == next)]
            next_ind = [x for x in next_ind if x > curr_ind]
            next_ind = next_ind[0]
            length = next_ind - curr_ind
            assert length >= 0
            n_points = length // stride
            if n_points < 1:
                desified_cnt.append(next)
            elif n_points < 2:
                desified_cnt.append(dense_cnt[curr_ind + length // 2])
                desified_cnt.append(next)
            else:
                for n in range(1, n_points):
                    desified_cnt.append(dense_cnt[curr_ind + n * (length // n_points)])
                if length - n_points * stride > 8:
                    desified_cnt.append(dense_cnt[curr_ind + n_points * stride])
                desified_cnt.append(next)
        desified_contours.append(desified_cnt)
    return desified_contours


def remove_extra_points(raw_contours, N):
    len_list = [len(ct) for ct in raw_contours]
    count = 0
    for i in range(len(len_list)):
        num = len_list[i]
        count += num
        if count >= N:
            return raw_contours[:(i - 1)]

    return raw_contours


def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2
    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i];

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp

def sample_line_points(raw_lines,init_stride=15,total_num=256):
    points_count = 1000
    lines = []
    if total_num:
        while points_count > total_num:
            lens = [len(ct) for ct in raw_lines]
            lens_stride = [l//init_stride for l in lens]
            if sum(lens_stride) <= total_num:
                stride = init_stride
            else:
                percentages = [num/sum(lens) for num in lens]
                num_points = [np.floor(i*total_num) for i in percentages]

                ids = [idx for idx in range(len(num_points)) if num_points[idx] < 3.0]
                num_points_extend = [max(3.0,i) for i in num_points]
                ex_num = sum([num_points_extend[i]-num_points[i] for i in range(len(num_points))])
                if len(ids):
                    sorted_id = sorted(range(len(num_points_extend)), key=lambda k: lens[k], reverse=True)
                    num_points_extend[sorted_id[0]] -= ex_num


                stride = [np.ceil(lens[i]/num_points_extend[i]) for i in range(len(lens))]


            lines = sample_line_points_without_N(raw_lines, init_stride=stride)
            lens2 = [len(ct) for ct in lines]
            points_count = sum(lens2)
            init_stride += 2
    else:
        lines = sample_line_points_without_N(raw_lines, init_stride=init_stride)


    return lines

def sample_coutours(raw_lines,stride=15):
    lines = []
    for i in range(len(raw_lines)):
        # for ps in contours:
        ps = raw_lines[i]

        stride_tmp = stride
        pre = [int(ps[0][0]), int(ps[0][1])]
        end = [int(ps[-1][0]), int(ps[-1][1])]

        n_points = len(ps) // stride_tmp
        line = []
        if n_points < 2:
            ind = len(ps) // 2
            midep = [int(ps[ind][0]), int(ps[ind][1])]
            line.append(pre)
            line.append(midep)
            line.append(end)
            continue
        for j in range(1, n_points + 2):
            if (j * stride_tmp >= len(ps)):
                line.append(pre)
                break

            p = [int(ps[j * stride_tmp][0]), int(ps[j * stride_tmp][1])]
            if (len(ps) - j * stride_tmp) <= stride_tmp:
                line.append(pre)
                line.append(end)
                break
            # elif (len(ps) - j * stride_tmp) <= stride_tmp:
            #     line.append(pre)
            #     line.append(p)
            #     line.append(end)
            #     break
            line.append(pre)
            pre = p
        lines.append(line)
    lens2 = [len(ct) for ct in lines]
    points_count = sum(lens2)

    return lines
def interpolrate_contours(contours, dense_contours, N=256, stride=20):
    len_list = [len(ct) for ct in contours]
    if sum(len_list) >= N:
        contours = remove_extra_points(contours, N)
        return contours
    elif N // 2 < sum(len_list) < N:
        percentages = [num / sum(len_list) for num in len_list]
        num_points = [np.floor(i * N) for i in percentages]

        ids = [idx for idx in range(len(num_points)) if num_points[idx] < 3.0]
        num_points_extend = [max(len_list[i], num_points[i]) for i in range(len(num_points))]
        ex_num = sum([num_points_extend[i] - num_points[i] for i in range(len(num_points))])
        if len(ids):
            sorted_id = sorted(range(len(num_points_extend)), key=lambda k: len_list[k], reverse=True)
            num_points_extend[sorted_id[0]] -= ex_num

        contours_dense = [uniformsample(contours[j], num_points_extend[j]) for j in range(len(num_points))]

        return contours_dense
    else:
        total_num = 1000
        while total_num > N:
            contours_dense = desify_contours(contours, dense_contours, stride=stride)
            lens2 = [len(ct) for ct in contours_dense]
            total_num = sum(lens2)
            stride += 2
        return contours_dense


def Data_collate_poly(batch):
    imgs = []
    ori_imgs = []
    heatmap = []
    mask = []
    boundary_mask = []
    polys = []
    names = []
    for im in batch:
        imgs.append(torch.from_numpy(im['img']))
        ori_imgs.append(torch.from_numpy(im['ori_img']))
        heatmap.append(torch.from_numpy(im['heatmap']).unsqueeze(0))
        mask.append(torch.from_numpy(im['mask']).unsqueeze(0))
        boundary_mask.append(torch.from_numpy(im['boundary_mask']).unsqueeze(0))
        polys.append(im['polys'])
        names.append(im['name'])

    img_collection = torch.stack(imgs, dim=0)
    ori_img_collection = torch.stack(ori_imgs, dim=0)
    heatmap_collection = torch.stack(heatmap, dim=0)
    mask_collection = torch.stack(mask, dim=0)
    boudary_mask_collection = torch.stack(boundary_mask, dim=0)

    sample = {'ori_images': ori_img_collection, 'images': img_collection, 'heatmaps': heatmap_collection,
              'masks': mask_collection, 'boundary_masks': boudary_mask_collection, 'PM_label': [], 'polys': polys,
              'names': names}
    return sample

def Data_collate_road(batch):
    # variables as tensor
    imgs = []
    ori_imgs = []
    heatmap = []
    skel_points = []
    junctions = []
    mask = []
    boundary_mask = []
    names = []
    for im in batch:
        imgs.append(torch.from_numpy(im['img']))
        ori_imgs.append(torch.from_numpy(im['ori_img']))
        heatmap.append(torch.from_numpy(im['heatmap']).unsqueeze(0))
        mask.append(torch.from_numpy(im['mask']).unsqueeze(0))
        boundary_mask.append(torch.from_numpy(im['boundary_mask']).unsqueeze(0))
        skel_points.append(im['skel_points'])
        junctions.append(im['junctions'])
        names.append(im['name'])

    img_collection = torch.stack(imgs, dim=0)
    ori_img_collection = torch.stack(ori_imgs, dim=0)
    heatmap_collection = torch.stack(heatmap, dim=0)
    mask_collection = torch.stack(mask, dim=0)
    boudary_mask_collection = torch.stack(boundary_mask, dim=0)

    sample = {'ori_images': ori_img_collection, 'images': img_collection, 'heatmaps': heatmap_collection,
              'skel_points': skel_points,'junctions': junctions,
              'masks': mask_collection, 'boundary_masks': boudary_mask_collection, 'PM_label': [], 'names': names}
    return sample