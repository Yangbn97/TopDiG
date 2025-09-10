# -*- coding: UTF-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import torch.utils.data as data
import numpy as np
import torch
import random
from data.Image_Fold import *
from utils.poly_utils import *
from utils.data_utils import *
from skimage import io





def read_data(filepath, mode='train'):
    image_path = os.path.join(filepath, 'image')
    label_path = os.path.join(filepath, 'binary_map')
    image_list = os.listdir(image_path)
    # if mode=='train':
    #     image_list = random.sample(image_list, 100)
    # else:
    #     image_list = random.sample(image_list, 40)
    img_lists = []
    lab_lists = []
    # pointmap_lists = []
    for i in range(len(image_list)):
        image_id = image_list[i]
        # image_id='GF2_PMS2__L1A0000788763-MSS2_0_9.tif'
        label_id = image_id

        image = os.path.join(image_path, image_id)
        label = os.path.join(label_path, label_id)

        img_lists.append(image)
        lab_lists.append(label)
    return img_lists, lab_lists



class Dataset_water(data.Dataset):

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
            hierarchy = hierarchy[0]
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

            contours = [measure.approximate_polygon(ct, 1) for ct in contours_cv]
            filter_list = [1 if len(c) > 2 else 0 for c in contours]
            contours = [contours[f] for f in range(len(filter_list)) if filter_list[f]]
            contours_cv = [contours_cv[f] for f in range(len(filter_list)) if filter_list[f]]

            stride = int((256 / self.N) * 15)
            contours = interpolrate_contours(contours, contours_cv, N=self.N, stride=stride)

            contours = [c for c in contours if len(c) > 2]
            lens = [len(ct) for ct in contours]
            sorted_id = sorted(range(len(lens)), key=lambda k: lens[k], reverse=True)
            contours = [contours[i] for i in sorted_id]

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
        mask = np.uint8(mask)

        boundary_mask = region2boundary(mask)
        boundary_mask = cv2.GaussianBlur(boundary_mask, ksize=(self.dilate_pixels, self.dilate_pixels), sigmaX=1,
                                         sigmaY=1)
        boundary_mask[boundary_mask > 0] = 1
        boundary_mask = np.uint8(boundary_mask)

        heatmap = generate_heatmap(pointmap, size=3)

        # for cnt in contours:
        #     for p in range(len(cnt)-1):
        #         mid = (cnt[p] + cnt[p + 1]) / 2
        #         cv2.line(ori_img, (int(cnt[p][1]), int(cnt[p][0])), (int(cnt[p + 1][1]), int(cnt[p + 1][0])),
        #                  (242, 203, 5), thickness=2)
        #         cv2.arrowedLine(ori_img, (int(cnt[p][1]), int(cnt[p][0])), (int(mid[1]), int(mid[0])),
        #                         (242, 203, 5), thickness=2, tipLength=0.3)
        #         cv2.circle(ori_img, (int(cnt[p][1]), int(cnt[p][0])), 4, (5, 203, 242), -1)
        #         cv2.circle(ori_img, (int(cnt[p + 1][1]), int(cnt[p + 1][0])), 4, (5, 203, 242), -1)
        #
        # plt.figure()
        # plt.subplot(221)
        # plt.imshow(ori_img)
        # plt.title('image')
        # plt.axis('off')
        # plt.subplot(222)
        # plt.title('label')
        # plt.imshow(np.uint8(mask))
        # plt.axis('off')
        # plt.subplot(223)
        # plt.title('heat')
        # plt.imshow(heatmap)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

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

    # trainset = Dataset_water(r'/data02/ybn/Datasets/water/GID/cropped512/train', mode='train',N=320)
    trainset = Dataset_water(r'G:\Datasets\WaterDatasets\water-body-satellite-data\train', mode='train', N=320)
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