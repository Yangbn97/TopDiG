import os
import time
import numpy as np
import cv2
# import gdal, ogr, osr
import networkx as nx
from skimage.draw import polygon2mask
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

# def poly2shp(img_path,strShpFile,polys):
#     ds = gdal.Open(img_path)
#     geoTransforms = ds.GetGeoTransform()

#     gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")  # allow Chinese
#     gdal.SetConfigOption("SHAPE_ENCODING", "CP936")  # allow Attribute Table to support Chinese
#     ogr.RegisterAll()  # register all drivers

#     strDriverName = "ESRI Shapefile"  # create ESRI shapefile
#     oDriver = ogr.GetDriverByName(strDriverName)
#     if oDriver == None:
#         print('%s Driver not available' % strDriverName)
#         return
#     oDS = oDriver.CreateDataSource(strShpFile)
#     if oDS == None:
#         print("Fail to create file [%s]！", strShpFile)
#         return

#     srs = osr.SpatialReference()
#     papszLCO = []
#     # create Layer,"TestPolygon"->name of Attribute Table
#     oLayer = oDS.CreateLayer("TestPolygon", srs, ogr.wkbPolygon, papszLCO)
#     if oLayer == None:
#         print("Fail to create Layer！\n")
#         return

#     '''Add vector data, Attribute Table data, vector coordinates'''
#     oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)  # create FieldID
#     oLayer.CreateField(oFieldID, 1)
#     oDefn = oLayer.GetLayerDefn()  # define feature
#     i = 0

#     if len(polys) != 0:
#         for contour in polys:
#             if not np.array_equal(contour[0], contour[-1]):
#                 contour.append(contour[0])
#             box = ogr.Geometry(ogr.wkbLinearRing)
#             for point in contour:
#                 x_col, y_row = float(point[1]), float(point[0])
#                 x_col, y_row = pixelCoor2PlaneCoor(geoTransforms, (x_col, y_row))
#                 box.AddPoint(x_col, y_row)
#             oFeatureTriangle = ogr.Feature(oDefn)
#             oFeatureTriangle.SetField(0, i)
#             i += 1
#             garden = ogr.Geometry(ogr.wkbPolygon)  # define Polygon
#             garden.AddGeometry(box)  # contour to Polygon
#             gardens = ogr.Geometry(ogr.wkbMultiPolygon)  # define MultiPolygon
#             gardens.AddGeometry(garden)  # Polygon to MultiPolygon
#             gardens.CloseRings()
#             geomTriangle = ogr.CreateGeometryFromWkt(str(gardens))  # add closed Polygon to Attribute Table
#             oFeatureTriangle.SetGeometry(geomTriangle)
#             oLayer.CreateFeature(oFeatureTriangle)
#     oDS.Destroy()


# def LineGraph2shp(subgraphs, strShpFile, strImagePath):
#     os.makedirs(os.path.dirname(strShpFile), exist_ok=True)

#     ds = gdal.Open(strImagePath)
#     im_geotrans = ds.GetGeoTransform()

#     gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")  # allow Chinese
#     gdal.SetConfigOption("SHAPE_ENCODING", "CP936")  # allow Attribute Table to support Chinese
#     ogr.RegisterAll()  # register all drivers

#     strDriverName = "ESRI Shapefile"  # create ESRI shapefile
#     Driver = ogr.GetDriverByName(strDriverName)
#     if Driver == None:
#         print('%s Driver not available' % strDriverName)
#         return
#     DS = Driver.CreateDataSource(strShpFile)
#     if DS == None:
#         print("Fail to create file [%s]！", strShpFile)
#         return

#     srs = osr.SpatialReference()
#     papszLCO = []
#     # create Layer,"TestPolygon"->name of Attribute Table
#     oLayer = DS.CreateLayer("TestPolygon", srs, ogr.wkbMultiLineString, papszLCO)
#     if oLayer == None:
#         print("Fail to create Layer！\n")
#         return

#     '''Add vector data, Attribute Table data, vector coordinates'''
#     oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)  # create FieldID
#     oLayer.CreateField(oFieldID, 1)

#     oDefn = oLayer.GetLayerDefn()  # define feature
#     oFeatureTriangle = ogr.Feature(oDefn)
#     i = 0

#     multilines = ogr.Geometry(ogr.wkbMultiLineString)  # crate MultiLineString
#     for edge in nx.generate_edgelist(subgraphs, data=False):
#         line = ogr.Geometry(ogr.wkbLineString)  # crate LineString
#         edge = [int(n) for n in edge.split(' ')]
#         pt = subgraphs.nodes[edge[0]]['pos']
#         adj_pt = subgraphs.nodes[edge[1]]['pos']
#         for point in [pt, adj_pt]:
#             x_col, y_row = pixelCoor2PlaneCoor(im_geotrans, (float(point[1]), float(point[0])))  # coord conversion
#             line.AddPoint(x_col, y_row)  # add Point to LineString
#         multilines.AddGeometry(line)  # add LinString to MultiLineString
#     oFeatureTriangle.SetField(0, i)  # set field value, i.e. id of MultiLineString
#     i += 1
#     geomTriangle = ogr.CreateGeometryFromWkt(str(multilines))  # add MultiLineString to Attribute Table
#     oFeatureTriangle.SetGeometry(geomTriangle)
#     oLayer.CreateFeature(oFeatureTriangle)  # writhe MultiLineString to shapefile

#     DS.Destroy()


def poly2raster(img, save_path, polys):
    h, w, _ = img.shape
    raster_out = np.zeros((h, w), dtype=np.uint8)

    for contour in polys:
        if len(contour) > 0:
            if not np.array_equal(contour[0], contour[-1]):
                contour.append(contour[0])
            tmp = polygon2mask((h, w), contour)
            raster_out += tmp

    raster_out[raster_out > 0] = 255
    cv2.imwrite(save_path, raster_out)


def line2raster(img, save_path, subgraphs):
    h, w, _ = img.shape
    raster_out = np.zeros((h, w), dtype=np.uint8)
    for edge in nx.generate_edgelist(subgraphs, data=False):
        edge = [int(n) for n in edge.split(' ')]
        pt = subgraphs.nodes[edge[0]]['pos']
        adj_pt = subgraphs.nodes[edge[1]]['pos']
        cv2.line(raster_out, (int(pt[1]), int(pt[0])), (int(adj_pt[1]), int(adj_pt[0])), (255, 0, 0), 1)
    # raster_out = cv2.GaussianBlur(np.float32(raster_out), ksize=(5, 5), sigmaX=1, sigmaY=1)
    # raster_out[raster_out > 0] = 255
    # raster_out = np.uint8(raster_out)
    cv2.imwrite(save_path, raster_out)


