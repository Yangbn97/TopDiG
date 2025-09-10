import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.cluster.hierarchy import fclusterdata

def gpu_NMS(pointmap, score=None, N=None,gap=10):
    scoremap = pointmap.copy()
    # print(pointmap.shape)#(1024, 1024)
    pointmap = torch.from_numpy(pointmap)  # torch.Size([1024, 1024])
    pointmap = pointmap.cuda()
    pointmap = pointmap.unsqueeze(0)  # torch.Size([1, 1024, 1024])
    pointmap = pointmap.unsqueeze(0)  # torch.Size([1, 1, 1024, 1024])


    max_pointmap = F.max_pool2d(pointmap, kernel_size=9, stride=1, padding=4)  # torch.Size([1, 1, 1024, 1024])


    pointmap = pointmap[0, 0, :, :].data.cpu().numpy()
    max_pointmap = max_pointmap[0, 0, :, :].data.cpu().numpy()  # (1024, 1024)
    point_index = (max_pointmap == pointmap)  # (1024, 1024)

    pointmap = point_index * pointmap

    Nodes = []
    if N and not score:
        ro = 2
        while len(Nodes) < N:
            new_N = int(N*ro)
            index1d = np.argpartition(pointmap.ravel(), -new_N)[-new_N:]
            index2d = np.unravel_index(index1d, pointmap.shape)
            NMS_Points = index2d
            Nodes = zip(NMS_Points[0], NMS_Points[1])  # (x,y)
            Nodes = list(Nodes)  # list:313   [(394,0),(102,3),(496,3),(689,3)   ]
            Nodes = py_cpu_nms(Nodes, gap)
            ro += 0.5
        scores = [scoremap[int(pt[0])][int(pt[1])] for pt in Nodes]
        sorted_id = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        Nodes = np.stack([Nodes[i] for i in sorted_id])
        Nodes = Nodes[:N]

        return Nodes

    if score and not N:
        NMS_Points = np.where(pointmap >= score)  # 313 tuple:2  0={ndarry:(313.)}
        Nodes = zip(NMS_Points[0], NMS_Points[1])  # (x,y)
        Nodes = list(Nodes)  # list:313   [(394,0),(102,3),(496,3),(689,3)   ]
        Nodes = py_cpu_nms(Nodes, gap)
        return Nodes

    if score and N:
        node_len = 0
        while node_len == 0:
            NMS_Points = np.where(pointmap >= score)  # 313 tuple:2  0={ndarry:(313.)}
            Nodes = zip(NMS_Points[0], NMS_Points[1])  # (x,y)
            Nodes = list(Nodes)  # list:313   [(394,0),(102,3),(496,3),(689,3)   ]
            Nodes = py_cpu_nms(Nodes, gap)
            node_len = len(Nodes)
            score -= 0.5

        if node_len < N:
            repit_num = N // node_len
            Nodes = repit_num * Nodes
            ex = N - len(Nodes)
            Nodes.extend(Nodes[:ex])
        else:
            scores = [scoremap[int(pt[0])][int(pt[1])] for pt in Nodes]
            sorted_id = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            Nodes = np.stack([Nodes[i] for i in sorted_id])
            Nodes = Nodes[:N]
            node_len = N
        return Nodes,node_len



    if not N and not score:
        print('Must provide N or score!')
        exit(1)




def NMS(output):
    points = []
    point_map = np.zeros(output.shape, np.uint8)
    h, w = output.shape
    for i in range(0, h):
        for j in range(0, w):
            if (output[i][j] > 40):
                max = 0
                for m in range(-10, 10):
                    for n in range(-10, 10):
                        y = i + m
                        x = j + n
                        if (x >= 0 and x < w and y >= 0 and y < h):
                            if (max < (output[y][x])):
                                max = output[y][x]

                if (output[i][j] >= (max)):
                    points.append([float(j), float(i)])

    NMS_Point = py_cpu_nms(points, 10)
    for i in range(0, len(NMS_Point)):
        x = int(NMS_Point[i][0])
        y = int(NMS_Point[i][1])
        point_map[y][x] = 255
    return point_map, NMS_Point

def py_cpu_nms(points, thresh):
    """Pure Python NMS baseline."""
    # (313, 2)[[394 0],[102 3],[469 3]   ]
    points = np.array(points)

    if (len(points) > 0):
        x = np.array(points[:, 0])  # (313,)
        y = np.array(points[:, 1])  # (313,)
        order = np.array(range(0, len(x)))  # (313,)[0 1 2 3 4 5 6 7 8 9   ]

        NMS_Point = []
        while len(order) > 0:
            i = order[0]
            x0 = x[i]
            y0 = y[i]

            NMS_Point.append([x0, y0])  # [[394,0],[102,3],[496,3]     ]
            x1 = x[order[1:]]
            y1 = y[order[1:]]
            Dis = np.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))  # (313,)(312,)decresce gradually
            indes = np.where(Dis > thresh)[0]
            order = order[indes + 1]

        return NMS_Point

    if (len(points) <= 0):
        return []


