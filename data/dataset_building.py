# -*- coding: UTF-8 -*-
import os

import cv2
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.utils.data as data
import cv2 as cv
import numpy as np
import torch
from skimage import morphology, measure
import random
from glob import glob
from data.Image_Fold import default_loader,Color_Augment
from utils.poly_utils import *
from utils.data_utils import *
from skimage import io



def read_data(filepath, mode='train'):
    image_path = os.path.join(filepath, 'image')
    label_path = os.path.join(filepath, 'binary_map')
    mask_list = os.listdir(label_path)
    # if mode=='train':
    #     total_num = len(image_list)
    #     sample_num = int(0.05*total_num)
    #     image_list = image_list[:sample_num]
    # else:
    #     image_list = image_list

    
    img_lists = []
    lab_lists = []
    # pointmap_lists = []
    for i in range(len(mask_list)):
        label_id = mask_list[i]
        # image_id ='austin1_1_0.tif'
        label = os.path.join(label_path, label_id)
        if not os.path.exists(label):
                    label = os.path.join(label_path, label_id[:-4]+'.tif')

        image_id = label_id
        image = os.path.join(image_path, image_id)
        if not os.path.exists(image):
            image = os.path.join(image_path, image_id[:-4]+'.jpg')
        
        img_lists.append(image)
        lab_lists.append(label)
    return img_lists, lab_lists



class Dataset_Inria(data.Dataset):

    def __init__(self, ROOT, mode='train', N=600, dilate=5):
        imglists, labellists = read_data(ROOT, mode)
        self.img_list = imglists
        self.mask_list = labellists
        self.N = N
        self.dilate_pixels = dilate
        self.mode = mode


    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = self.mask_list[index]
        basename = os.path.basename(img_path)
        name = basename.split('.')[0]

        img = io.imread(img_path)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)


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
                stride = int((256 / self.N) * 20)
                contours = interpolrate_contours(contours, contours_cv, N=self.N, stride=stride)

            contours = [c for c in contours if len(c) > 2]
            lens = [len(ct) for ct in contours]
            sorted_id = sorted(range(len(lens)), key=lambda k: lens[k], reverse=True)
            contours = [contours[i] for i in sorted_id]

            if sum(lens) > self.N:
                contours = remove_extra_points(contours, N=self.N)

        else:
            contours = contours_cv

        # for cnt in contours:
        #     for p in range(len(cnt)):
        #         if p == len(cnt) - 1:
        #             mid = (cnt[p] + cnt[0]) / 2
        #             cv2.line(ori_img, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[0][1]), int(cnt[0][0])),
        #                      (242, 203, 5), 2)
        #             cv2.arrowedLine(ori_img, (int(cnt[p][1]), int(cnt[p][0])), (int(mid[1]), int(mid[0])),
        #                             (242, 203, 5), thickness=2, tipLength=0.3)
        #             cv2.circle(ori_img, (int(cnt[p][1]), int(cnt[p][0])), 4, (5, 203, 242), -1)
        #             cv2.circle(ori_img, (int(cnt[0][1]), int(cnt[0][0])), 4, (5, 203, 242), -1)
        #
        #         else:
        #             mid = (cnt[p] + cnt[p + 1]) / 2
        #             cv2.line(ori_img, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[p + 1][1]), int(cnt[p + 1][0])),
        #                      (242, 203, 5), thickness=2)
        #             cv2.arrowedLine(ori_img, (int(cnt[p][1]), int(cnt[p][0])), (int(mid[1]), int(mid[0])),
        #                             (242, 203, 5), thickness=2, tipLength=0.3)
        #             cv2.circle(ori_img, (int(cnt[p][1]), int(cnt[p][0])), 4, (5, 203, 242), -1)
        #             cv2.circle(ori_img, (int(cnt[p + 1][1]), int(cnt[p + 1][0])), 4, (5, 203, 242), -1)
        #
        # cv2.imwrite(r'./building_good.png',ori_img)
        # plt.imshow(ori_img)
        # plt.show()

        pointmap = np.zeros_like(mask)
        for ct in contours:
            for p in ct:
                pointmap[int(p[0])][int(p[1])] = 255


        img = np.array(img, np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        mask[mask > 0] = 1
        mask[mask <= 0] = 0
        mask = np.uint8(mask)

        boundary_mask = region2boundary(mask)
        boundary_mask = cv2.GaussianBlur(boundary_mask, ksize=(self.dilate_pixels, self.dilate_pixels), sigmaX=1, sigmaY=1)
        boundary_mask[boundary_mask > 0] = 1
        boundary_mask = np.uint8(boundary_mask)


        heatmap = generate_heatmap(pointmap,size=3)


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

    # trainset = Dataset_Inria('/data02/ybn/Datasets/Building/Inria/cropped300/train', mode='train')
    trainset = Dataset_Inria(r'G:\Datasets\BuildingDatasets\AerialImageDataset\cropped300\valid',mode='val',N=256)
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
