# -*- coding: UTF-8 -*-
import os

import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.utils.data as data
import cv2 as cv
import numpy as np
import torch
from skimage import morphology, measure,io
import random
from glob import glob
from data.Image_Fold import default_loader, Color_Augment
from utils.poly_utils import *
from utils.data_utils import *



def read_data(filepath, mode='train'):
    image_path = os.path.join(filepath, 'image')
    label_path = os.path.join(filepath, 'binary_map')
    image_list = os.listdir(image_path)

    img_lists = []
    lab_lists = []
    for i in range(len(image_list)):
        image_id = image_list[i]
        # image_id = '10828780_15_2_2.tif'
        # if mode == 'valid':
        #     image_id='10978795_15_0_3.tif'
        label_id = image_id

        image = os.path.join(image_path, image_id)
        label = os.path.join(label_path, label_id)

        img_lists.append(image)
        lab_lists.append(label)
    return img_lists, lab_lists


class Dataset_road(data.Dataset):

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


        keypoints = np.array([])

        ori_img = img.copy()
        if self.mode == 'train':
            rand = np.random.random(3)
            img = default_loader(img, rand)
            mask = default_loader(mask, rand)
            ori_img = img.copy()
            rand2 = np.random.random(2)
            img = Color_Augment(img, rand2)



        stride = int((256 / self.N) * 15)
        keypoints, junction_nodes = sample_road(img, mask, N=self.N, init_stride=stride)

        # keypoints = np.asarray([np.asarray(t) for t in keypoints])
        # for cnt in keypoints:
        #     for p in range(len(cnt)-1):
        #         mid = (cnt[p] + cnt[p + 1]) / 2
        #         cv2.line(ori_img, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[p + 1][1]), int(cnt[p + 1][0])),
        #                  (242, 203, 5), thickness=2)
        #         cv2.arrowedLine(ori_img, (int(cnt[p][1]), int(cnt[p][0])), (int(mid[1]), int(mid[0])),
        #                         (242, 203, 5), thickness=2, tipLength=0.3)
        #         cv2.circle(ori_img, (int(cnt[p][1]), int(cnt[p][0])), 4, (5, 203, 242), -1)
        #         cv2.circle(ori_img, (int(cnt[p + 1][1]), int(cnt[p + 1][0])), 4, (5, 203, 242), -1)
        #
        # cv2.imwrite(r'./road_good.png', ori_img)
        # plt.imshow(ori_img)
        # plt.show()



        pointmap = np.zeros_like(mask)
        for l in range(len(keypoints)):
            for p in range(len(keypoints[l])):
                pt = keypoints[l][p]
                pointmap[int(pt[0])][int(pt[1])] = 255

        img = np.array(img, np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        mask[mask > 0] = 1
        mask[mask <= 0] = 0

        boundary_mask = np.uint8(skeletonize(mask)* 255)
        boundary_mask = cv2.GaussianBlur(boundary_mask, ksize=(self.dilate_pixels, self.dilate_pixels), sigmaX=1,
                                         sigmaY=1)
        boundary_mask[boundary_mask > 0] = 1
        boundary_mask = np.uint8(boundary_mask)

        heatmap = generate_heatmap(pointmap, size=3)

        # plt.figure()
        # plt.subplot(221)
        # plt.imshow(ori_img)
        # plt.title('img')
        # plt.subplot(222)
        # plt.imshow(mask)
        # plt.title('GT')
        # plt.subplot(223)
        # plt.imshow(boundary_mask)
        # plt.title('bdy GT')
        # plt.subplot(224)
        # plt.imshow(heatmap)
        # plt.title('Heatmap GT')
        # plt.savefig('data_check.png', bbox_inches='tight')

        batch = {
            'ori_img': ori_img,
            'img': img,
            'heatmap': heatmap,
            'mask': mask,
            'boundary_mask': boundary_mask,
            'skel_points': keypoints,
            'junctions':junction_nodes,
            'name': name
        }

        return batch

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    import os

    # Dataset = Dataset_road(r'/data02/ybn/Datasets/Road/Massachusetts/cropped300/train', mode='valid', N=256)
    Dataset = Dataset_road(r'G:\Datasets\RoadDatasets\Massachusetts\cropped300\val', mode='valid', N=256)
    loader = torch.utils.data.DataLoader(
        Dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=Data_collate_road)
    for i, batch in enumerate(loader):
        print(i)
        # pointmap, anglemap, keypoints = get_topoData(mask, img)
        # print(idx, img.shape)
