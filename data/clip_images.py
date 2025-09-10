import cv2
import numpy as np
import cv2 as cv
import glob
import os
from skimage import morphology, io
import gdal
from tqdm import tqdm

refer_dir = r'G:\Datasets\GID\water\cropped512\nozero_valid\binary_map'


def read_tif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
        return
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    if im_bands == 1:
        return im_data
    im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()  # 获取投影信息

    # print('im_proj: ', im_proj)

    # print('im_geotrans: ', im_geotrans)

    data = im_data[0:3, 0:im_height, 0:im_width]
    data = np.transpose(data, (1, 2, 0))
    data = data[:, :, ::-1]

    return data


def make_Patch(Stride, Patch_Size, Image_Path, Lable_Path, Image_Save_Path, Label_Save_Path):
    i = 1

    imgLists = glob.glob(Image_Path)
    label_patch_Lists = os.listdir(Label_Save_Path)
    for imgpath in tqdm(imgLists):
        # img = cv.imread(imgpath)
        img = read_tif(imgpath)
        h, w, _ = img.shape
        basename = os.path.basename(imgpath).split('.')[0]
        labelpath = os.path.join(Lable_Path, '{}.tif'.format(basename))
        label = cv.imread(labelpath, 0)
        for y in range(0, (h - Patch_Size) // Stride + 2):
            for x in range(0, (w - Patch_Size) // Stride + 2):
                basename_patch = basename + '_' + str(y) + '_' + str(x) + '.tif'
                img_save = os.path.join(Image_Save_Path, basename + '_' + str(y) + '_' + str(x) + '.tif')
                label_save = os.path.join(Label_Save_Path, basename + '_' + str(y) + '_' + str(x) + '.tif')
                if basename_patch in label_patch_Lists:
                    continue
                if (y * Stride + Patch_Size <= h and x * Stride + Patch_Size <= w):
                    Patch = img[y * Stride:y * Stride + Patch_Size, x * Stride:x * Stride + Patch_Size]
                    labelPatch = label[y * Stride:y * Stride + Patch_Size, x * Stride:x * Stride + Patch_Size]
                    if np.sum(labelPatch) > 0 and len(np.unique(Patch)) > 1:
                        cv.imwrite(img_save, Patch)
                        cv.imwrite(label_save, labelPatch)

                if (y * Stride + Patch_Size <= h and x * Stride < w and x * Stride + Patch_Size > w):
                    Patch = img[y * Stride:y * Stride + Patch_Size, w - Patch_Size:w]
                    labelPatch = label[y * Stride:y * Stride + Patch_Size, w - Patch_Size:w]
                    if np.sum(labelPatch) > 0 and len(np.unique(Patch)) > 1:
                        cv.imwrite(img_save, Patch)
                        cv.imwrite(label_save, labelPatch)

                if (y * Stride < h and y * Stride + Patch_Size > h and x * Stride + Patch_Size <= w):
                    Patch = img[h - Patch_Size:h, x * Stride:x * Stride + Patch_Size]
                    labelPatch = label[h - Patch_Size:h, x * Stride:x * Stride + Patch_Size]
                    if np.sum(labelPatch) > 0 and len(np.unique(Patch)) > 1:
                        cv.imwrite(img_save, Patch)
                        cv.imwrite(label_save, labelPatch)

                if (y * Stride < h and y * Stride + Patch_Size > h and x * Stride < w and x * Stride + Patch_Size > w):
                    Patch = img[h - Patch_Size:h, w - Patch_Size:w]
                    labelPatch = label[h - Patch_Size:h, w - Patch_Size:w]
                    if np.sum(labelPatch) > 0 and len(np.unique(Patch)) > 1:
                        cv.imwrite(img_save, Patch)
                        cv.imwrite(label_save, labelPatch)

            # print(i)
            i = i + 1

def make_Patch_from_refer(Stride, Patch_Size, Label_Path, Label_Save_Path):
    i = 1

    imgLists = glob.glob(Label_Path)
    refer_list = os.listdir(refer_dir)
    # refer_list = [f.split('.')[0] for f in refer_list]
    # refer_list = [b.split('_')[-2:] for b in refer_list]
    # refer_list = [[int(n[0]),int(n[1])] for n in refer_list]
    for imgpath in tqdm(imgLists):
        # img = cv.imread(imgpath)
        img = read_tif(imgpath)
        h, w, _ = img.shape
        basename = os.path.basename(imgpath).split('.')[0].replace('_label','')

        for y in range(0, (h - Patch_Size) // Stride + 2):
            for x in range(0, (w - Patch_Size) // Stride + 2):
                basename_patch = basename + '_' + str(y) + '_' + str(x) + '.tif'
                if basename_patch in refer_list:
                    label_save = os.path.join(Label_Save_Path, basename + '_' + str(y) + '_' + str(x) + '.tif')

                    if (y * Stride + Patch_Size <= h and x * Stride + Patch_Size <= w):
                        Patch = img[y * Stride:y * Stride + Patch_Size, x * Stride:x * Stride + Patch_Size]
                        cv.imwrite(label_save, Patch)

                    if (y * Stride + Patch_Size <= h and x * Stride < w and x * Stride + Patch_Size > w):
                        Patch = img[y * Stride:y * Stride + Patch_Size, w - Patch_Size:w]
                        cv.imwrite(label_save, Patch)

                    if (y * Stride < h and y * Stride + Patch_Size > h and x * Stride + Patch_Size <= w):
                        Patch = img[h - Patch_Size:h, x * Stride:x * Stride + Patch_Size]
                        cv.imwrite(label_save, Patch)

                    if (y * Stride < h and y * Stride + Patch_Size > h and x * Stride < w and x * Stride + Patch_Size > w):
                        Patch = img[h - Patch_Size:h, w - Patch_Size:w]
                        cv.imwrite(label_save, Patch)

            # print(i)
            i = i + 1

def compress_data(image_path):
    img_list = os.listdir(image_path)
    for img_filename in img_list:
        path = os.path.join(image_path, img_filename)
        img = io.imread(path, as_gray=True)
        img = np.array(img, dtype=np.uint8)
        cv.imwrite(path, img)


def MultiClass2SingleClass(label_path, label_save_path):
    img_list = os.listdir(label_path)
    for img_filename in tqdm(img_list):
        path = os.path.join(label_path, img_filename)
        save_path = os.path.join(label_save_path, img_filename)
        label = io.imread(path)
        # label[label != 1] = 0
        # label[label == 1] = 255
        mask_mapping = {
            (255, 0, 0): 0,
            (0, 255, 0): 1,
            (0, 255, 255): 2,
            (255, 255, 0): 3,
            (0, 0, 255): 4,
        }
        for k in mask_mapping:
            label[(label == k).all(axis=2)] = mask_mapping[k]
        label = label[:, :, 0]
        # label[label != 4] = 0
        # label[label == 4] = 255
        label = np.uint8(label)
        cv.imwrite(save_path, label)


if __name__ == '__main__':
    Stride = 512
    Patch_Size = 512
    Root = r'/data02/ybn/Datasets/Road/Massachusetts/raw/train'

    Image_Path = r'G:\Datasets\GID\Large-scale Classification_5classes\image_RGB/*.tif'
    Lable_Path = r'G:\Datasets\GID\Large-scale Classification_5classes\label_5classes/*.tif'
    Image_save_Path = r'/data02/ybn/Datasets/Road/Massachusetts/cropped320/train/image'
    label_save_path = r'G:\Datasets\GID\water\cropped512\nozero_valid\5classes_label'
    # MultiClass2SingleClass(Lable_Path,label_save_path)
    # compress_data(Lable_Path)
    # make_Patch(Stride, Patch_Size, Image_Path, Lable_Path, Image_save_Path, label_save_path)
    # make_Patch_from_refer(Stride,Patch_Size,Lable_Path,label_save_path)
    MultiClass2SingleClass(label_save_path,label_save_path)
