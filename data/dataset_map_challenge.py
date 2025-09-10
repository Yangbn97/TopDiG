# -*- coding: UTF-8 -*-
import os

import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.utils.data as data
import cv2 as cv
import numpy as np
import torch
from skimage import morphology, measure
import random
from glob import glob
from data.Image_Fold import default_loader, Color_Augment
from utils.poly_utils import *
from utils.data_utils import *

from pycocotools.coco import COCO
from skimage.draw import polygon2mask
from skimage import io




def read_data(Root, mode='train'):
    if mode == 'train':
        annFile = os.path.join(Root, 'annotation.json')
    else:
        annFile = os.path.join(Root, 'annotation-small.json')

    coco = COCO(annFile)
    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['building'])
    imgIds = coco.getImgIds(catIds=catIds)
    return imgIds, catIds, coco




class Dataset_MC(data.Dataset):

    def __init__(self, ROOT, mode='train', N=256, dilate=5):
        img_list, catIds, coco = read_data(ROOT, mode)
        self.img_list = img_list
        self.catIds = catIds
        self.coco = coco
        self.img_path = os.path.join(ROOT, 'image')
        self.N = N
        self.dilate_pixels = dilate
        self.mode = mode

    def __getitem__(self, index):

        imgId = self.img_list[index]
        img_dic = self.coco.loadImgs(imgId)[0]

        file_name = img_dic['file_name']

        file_path = os.path.join(self.img_path, file_name)
        annIds = self.coco.getAnnIds(imgIds=img_dic['id'], catIds=self.catIds, iscrowd=None)
        ann = self.coco.loadAnns(annIds)

        img = io.imread(file_path)

        basename = os.path.basename(file_path)
        name = basename.split('.')[0]
        instance_polys = [np.array(poly).reshape(-1, 2) for obj in ann for poly in obj['segmentation']]
        polys = [cnt[:, [-1, 0]] for cnt in instance_polys]
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for i in range(len(polys)):
            mask += polygon2mask((img.shape[0], img.shape[1]), polys[i])
        mask = np.uint8(mask) * 255


        ori_img = img.copy()

        if self.mode == 'train':
            rand = np.random.random(3)
            img = default_loader(img, rand)
            mask = default_loader(mask, rand)
            ori_img = img.copy()
            rand2 = np.random.random(2)
            img = Color_Augment(img, rand2)

        contours_cv, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_cv = [np.squeeze(cnt) for cnt in contours_cv]
        inds = [i for i, x in enumerate(contours_cv) if len(x) > 2]
        contours_cv = [contours_cv[i] for i in inds]
        if len(contours_cv):
            hierarchy = np.squeeze(hierarchy)
            hierarchy = hierarchy[None, :] if len(hierarchy.shape) == 1 else hierarchy
            hierarchy = [hierarchy[i] for i in inds]
            contours_cv = [c[:, ::-1][::-1] for c in contours_cv]

            for h in range(len(hierarchy)):
                if hierarchy[h][-1] > -1:
                    contours_cv[h] = contours_cv[h][::-1]

            contours_cv = [c for c in contours_cv if len(c) > 2]
            for i in range(len(contours_cv)):
                c = contours_cv[i]
                if np.sqrt((c[0][0] - c[-1][0]) ** 2 + (c[0][1] - c[-1][1]) ** 2) <= 2:
                    contours_cv[i][-1] = contours_cv[i][0]

            contours = [measure.approximate_polygon(ct, 3) for ct in contours_cv]

            ids = [j for j in range(len(contours)) if len(contours[j]) > 2]
            contours_cv = [contours_cv[j] for j in ids]
            contours = [contours[j] for j in ids]

            if self.N > 256:
                stride = int((256 / self.N)*20)
                contours = interpolrate_contours(contours, contours_cv, N=self.N,stride=stride)

            contours = [c for c in contours if len(c) > 2]
            lens = [len(ct) for ct in contours]
            sorted_id = sorted(range(len(lens)), key=lambda k: lens[k], reverse=True)
            contours = [np.asarray(contours[i]) for i in sorted_id]

            if sum(lens) > self.N:
                contours = remove_extra_points(contours, N=self.N)

        else:
            contours = contours_cv



        pointmap = np.zeros_like(mask)
        for ct in contours:
            for p in ct:
                pointmap[int(p[0])][int(p[1])] = 255


        img = np.array(img, np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        mask[mask > 0] = 1
        mask[mask <= 0] = 0

        boundary_mask = region2boundary(mask)
        boundary_mask = cv2.GaussianBlur(boundary_mask, ksize=(self.dilate_pixels, self.dilate_pixels), sigmaX=1, sigmaY=1)
        boundary_mask[boundary_mask > 0] = 1
        boundary_mask = np.uint8(boundary_mask)

        heatmap = generate_heatmap(pointmap, size=3)

        batch = {
            'ori_img': ori_img,
            'img': img,
            'heatmap': heatmap,
            'mask': mask,
            'boundary_mask': boundary_mask,
            'polys': contours,
            'name': name
        }

        return batch

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    import os

    trainset = Dataset_MC('/data01/ybn/Datasets/Building_Dateset/mapchallenge/val', mode='val')
    # trainset = Dataset_MC(r'G:\Datasets\BuildingDatasets\mapchallenge\val', mode='val')
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=Data_collate_poly, pin_memory=True)
    for i, batch in enumerate(train_loader):
        print(i)
        # pointmap, anglemap, keypoints = get_topoData(mask, img)
        # print(idx, img.shape)

# pre_path1 = '/home/wyj/Downloads/project/road_extraction/submits/our_s_30/'
# pre_path2 = '/home/wyj/Downloads/project/road_extraction/submits/ours_s_30_seg/'
#
# files = os.listdir(pre_path1)
#
# for file in files:
#     pre_img1 = np.array(cv.imread(pre_path1 + file ,cv.IMREAD_GRAYSCALE),np.float32)
#     pre_img2 = np.array(cv.imread(pre_path2 + file, cv.IMREAD_GRAYSCALE),np.float32)
#
#     pre_img1[pre_img1 <=128] = 0
#     pre_img1[pre_img1 >128] = 1
#     pre_img2[pre_img2 <= 128] = 0
#     pre_img2[pre_img2 > 128] = 1
#
#     pre_img = (pre_img1 + pre_img2) / 2
#     pre_img[pre_img>=0.5] = 255
#     pre_img[pre_img<0.5] = 0
#     cv.imwrite('/home/wyj/Downloads/project/road_extraction/submits/ours_s_30_fusion/'+file, np.array(pre_img, np.uint8))
