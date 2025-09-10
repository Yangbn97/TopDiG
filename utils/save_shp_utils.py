import os
import time
from skimage import measure
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import numpy as np
import cv2
import gdal, ogr, osr
import networkx as nx
from sknw import build_sknw
from glob import glob
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings('ignore')


def pixelCoor2PlaneCoor(geoTransforms, coor):
    '''
    :param geoTransforms: x0, dx, dxdy, y0, dydx, dy
    :param coor: pixel coor
    :return: plane coor
    '''
    x0, dx, dxdy, y0, dydx, dy = geoTransforms
    if dy > 0:
        dy = - dy
    x = x0 + dx * coor[0]
    y = y0 + dy * coor[1]
    return (x, y)

def is_hollow(coor0, coorn):
    """
    :param coor0: x-min, x-max, y-min, y-max (large)
    :param coorn: x-min, x-max, y-min, y-max
    :return: True/False
    """
    return (coorn[0]>coor0[0] and coorn[1]<coor0[1]) and (coorn[2]>coor0[2] and coorn[3]<coor0[3])


def poly2box(poly):
    """
    :param poly: Numpy, (N, 2)
    :return: x-min, x-max, y-min, y-max
    """
    x_min, x_max = min(poly[:, 0]), max(poly[:, 0])
    y_min, y_max = min(poly[:, 1]), max(poly[:, 1])
    return (x_min, x_max, y_min, y_max)

def poly2shp(img_path,save_path,polys):
    ds = gdal.Open(img_path)
    geoTransforms = ds.GetGeoTransform()
    strShpFile = os.path.join(save_path, os.path.basename(img_path)[:-4] + '.shp')
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")  # 为了支持中文路径
    gdal.SetConfigOption("SHAPE_ENCODING", "CP936")  # 为了使属性表字段支持中文
    ogr.RegisterAll()  # 注册所有驱动
    strDriverName = "ESRI Shapefile"  # 创建ESRI的shp文件
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        print('%s 驱动不可用' % strDriverName)
        return

    oDS = oDriver.CreateDataSource(strShpFile)  # 创建数据源
    if oDS == None:
        print("创建文件[%s]失败！", strShpFile)
        return
    srs = osr.SpatialReference()
    papszLCO = []
    # 创建图层，创建多边形图层,"TestPolygon"->属性表名
    oLayer = oDS.CreateLayer("TestPolygon", srs, ogr.wkbPolygon, papszLCO)
    if oLayer == None:
        print("图层创建失败！\n")
        return
    '''下面添加矢量数据，属性表数据、矢量数据坐标'''
    oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)  # 创建FieldID的整型属性
    oLayer.CreateField(oFieldID, 1)
    oDefn = oLayer.GetLayerDefn()  # 定义要素
    i = 0

    if len(polys) != 0:
        for contour in polys:
            if not np.array_equal(contour[0], contour[-1]):
                contour.append(contour[0])
            box = ogr.Geometry(ogr.wkbLinearRing)
            for point in contour:
                x_col, y_row = float(point[1]), float(point[0])
                x_col, y_row = pixelCoor2PlaneCoor(geoTransforms, (x_col, y_row))
                box.AddPoint(x_col, - y_row)
            oFeatureTriangle = ogr.Feature(oDefn)
            oFeatureTriangle.SetField(0, i)
            i += 1
            garden = ogr.Geometry(ogr.wkbPolygon)  # 每次重新定义多边形
            garden.AddGeometry(box)  # 将轮廓坐标放在单多边形中
            gardens = ogr.Geometry(ogr.wkbMultiPolygon)  # 每次重新定义多个多边形
            gardens.AddGeometry(garden)  # 依次将单多边形放入总的多边形集中
            gardens.CloseRings()
            geomTriangle = ogr.CreateGeometryFromWkt(str(gardens))  # 将封闭后的多边形集添加到属性表
            oFeatureTriangle.SetGeometry(geomTriangle)
            oLayer.CreateFeature(oFeatureTriangle)
    oDS.Destroy()

    # oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)  # 创建FieldID的整型属性
    # oLayer.CreateField(oFieldID, 1)
    # oDefn = oLayer.GetLayerDefn()  # 定义要素
    # oFeatureTriangle = ogr.Feature(oDefn)
    # i = 0
    # polygons = [Polygon(poly) for poly in polys]
    # tmp_poly = []
    #
    # for polygon in polygons:
    #     contour = np.array(polygon.exterior.coords)
    #     coor = poly2box(contour)
    #     box = ogr.Geometry(ogr.wkbLinearRing)
    #     for point in contour:
    #         # box.AddPoint(pixelCoor2PlaneCoor(im_geotrans, (float(point[0]), float(point[1]))))
    #         x_col, y_row = pixelCoor2PlaneCoor(geoTransforms, (float(point[0]), float(point[1])))
    #         box.AddPoint(x_col, y_row)
    #     if len(tmp_poly) != 0 and not is_hollow(tmp_poly[0][0], coor):
    #         gardens = ogr.Geometry(ogr.wkbMultiPolygon)  # 每次重新定义多个多边形
    #         for tm_poly in tmp_poly:
    #             box1 = tm_poly[1]
    #             oFeatureTriangle.SetField(0, i)
    #             garden = ogr.Geometry(ogr.wkbPolygon)  # 每次重新定义多边形
    #             garden.AddGeometry(box1)  # 将轮廓坐标放在单多边形中
    #             gardens.AddGeometry(garden)  # 依次将单多边形放入总的多边形集中
    #         gardens.CloseRings()
    #         geomTriangle = ogr.CreateGeometryFromWkt(str(gardens))  # 将封闭后的多边形集添加到属性表
    #         oFeatureTriangle.SetGeometry(geomTriangle)
    #         oLayer.CreateFeature(oFeatureTriangle)
    #         tmp_poly = []
    #         i += 1
    #     tmp_poly.append((coor, box))
    #
    # if len(tmp_poly) != 0:
    #     gardens = ogr.Geometry(ogr.wkbMultiPolygon)  # 每次重新定义多个多边形
    #     for tm_poly in tmp_poly:
    #         box1 = tm_poly[1]
    #         oFeatureTriangle.SetField(0, i)
    #         garden = ogr.Geometry(ogr.wkbPolygon)  # 每次重新定义多边形
    #         garden.AddGeometry(box1)  # 将轮廓坐标放在单多边形中
    #         gardens.AddGeometry(garden)  # 依次将单多边形放入总的多边形集中
    #     gardens.CloseRings()
    #     geomTriangle = ogr.CreateGeometryFromWkt(str(gardens))  # 将封闭后的多边形集添加到属性表
    #     oFeatureTriangle.SetGeometry(geomTriangle)
    #     oLayer.CreateFeature(oFeatureTriangle)
    # else:
    #     gardens = ogr.Geometry(ogr.wkbMultiPolygon)
    #     gardens.CloseRings()
    #     geomTriangle = ogr.CreateGeometryFromWkt(str(gardens))  # 将封闭后的多边形集添加到属性表
    #     oFeatureTriangle.SetGeometry(geomTriangle)
    #     oLayer.CreateFeature(oFeatureTriangle)
    #
    # oDS.Destroy()


def LineGraph2shp(subgraphs, strShpFile, strImagePath):
    os.makedirs(os.path.dirname(strShpFile), exist_ok=True)

    ds = gdal.Open(strImagePath)
    im_geotrans = ds.GetGeoTransform()

    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")  # 为了支持中文路径
    gdal.SetConfigOption("SHAPE_ENCODING", "CP936")  # 为了使属性表字段支持中文
    ogr.RegisterAll()  # 注册所有驱动

    strDriverName = "ESRI Shapefile"  # 创建ESRI的shp文件
    Driver = ogr.GetDriverByName(strDriverName)
    if Driver == None:
        print('%s 驱动不可用' % strDriverName)
        return
    DS = Driver.CreateDataSource(strShpFile)  # 创建数据源
    if DS == None:
        print("创建文件[%s]失败！", strShpFile)
        return

    srs = osr.SpatialReference()
    papszLCO = []
    # 创建图层，创建多边形图层,"TestPolygon"->属性表名
    oLayer = DS.CreateLayer("TestPolygon", srs, ogr.wkbMultiLineString, papszLCO)
    if oLayer == None:
        print("图层创建失败！\n")
        return

    '''下面添加矢量数据，属性表数据、矢量数据坐标'''
    oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)  # 创建FieldID的整型属性
    oLayer.CreateField(oFieldID, 1)

    oDefn = oLayer.GetLayerDefn()  # 定义要素
    oFeatureTriangle = ogr.Feature(oDefn)
    i = 0

    # for subgraph in subgraphs:
    #     multilines = ogr.Geometry(ogr.wkbMultiLineString)  # 创建MultiLineString类型的要素
    #     for edge in nx.generate_edgelist(subgraph, data=False):
    #         line = ogr.Geometry(ogr.wkbLineString)  # 创建单线要素
    #         edge = [int(n) for n in edge.split(' ')]
    #         pt = subgraph.nodes[edge[0]]['pos']
    #         adj_pt = subgraph.nodes[edge[1]]['pos']
    #         for point in [pt, adj_pt]:
    #             x_col, y_row = pixelCoor2PlaneCoor(im_geotrans, (float(point[1]), float(point[0])))  # 像素坐标转平面坐标
    #             line.AddPoint(x_col, y_row)  # 把点添加进单线要素
    #         multilines.AddGeometry(line)  # 把单线要素添加进多线要素
    #     oFeatureTriangle.SetField(0, i)  # 设置field value,即多线要素的id
    #     i += 1
    #     geomTriangle = ogr.CreateGeometryFromWkt(str(multilines))  # 将多线的点集添加到属性表
    #     oFeatureTriangle.SetGeometry(geomTriangle)
    #     oLayer.CreateFeature(oFeatureTriangle)  # 把该多线要素写入shapefile

    multilines = ogr.Geometry(ogr.wkbMultiLineString)  # 创建MultiLineString类型的要素
    for edge in nx.generate_edgelist(subgraphs, data=False):
        line = ogr.Geometry(ogr.wkbLineString)  # 创建单线要素
        edge = [int(n) for n in edge.split(' ')]
        pt = subgraphs.nodes[edge[0]]['pos']
        adj_pt = subgraphs.nodes[edge[1]]['pos']
        for point in [pt, adj_pt]:
            x_col, y_row = pixelCoor2PlaneCoor(im_geotrans, (float(point[1]), float(point[0])))  # 像素坐标转平面坐标
            line.AddPoint(x_col, y_row)  # 把点添加进单线要素
        multilines.AddGeometry(line)  # 把单线要素添加进多线要素
    oFeatureTriangle.SetField(0, i)  # 设置field value,即多线要素的id
    i += 1
    geomTriangle = ogr.CreateGeometryFromWkt(str(multilines))  # 将多线的点集添加到属性表
    oFeatureTriangle.SetGeometry(geomTriangle)
    oLayer.CreateFeature(oFeatureTriangle)  # 把该多线要素写入shapefile

    DS.Destroy()