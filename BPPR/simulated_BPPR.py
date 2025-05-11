import math
import sys
import csv
import datetime
from pyproj import Geod
import os
import numpy as np
from pyspark import SparkConf, SparkContext
import geopandas as gpd
import pandas as pd
import pygplates
import logging
from rotation_utils import get_reconstruction_model_dict


# ----- tool functions -----
# read points from csv/xlsx/xls/txt/shp
def read_points(file_path):
    return {
        'csv': lambda: read_points_from_csv(file_path),
        'xlsx': lambda: read_points_from_excel(file_path),
        'xls': lambda: read_points_from_excel(file_path),
        'txt': lambda: read_points_from_txt(file_path),
        'shp': lambda: read_points_from_shp(file_path)
    }.get(file_path.split(".")[1].lower(), lambda: None)()


# read points from csv
def read_points_from_csv(file_path):
    # decide if use df.drop() by data
    df = pd.read_csv(file_path)
    # df.drop(df.columns[0], axis=1, inplace=True)
    return df


# read points from excel
def read_points_from_excel(file_path):
    df = pd.read_excel(file_path)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df


# read points from txt
def read_points_from_txt(file_path):
    df = pd.read_table(file_path, delimiter=",")
    df.drop(df.columns[0], axis=1, inplace=True)
    return df


# read points from shp
def read_points_from_shp(file_path):
    df = gpd.read_file(file_path, driver='SHP')
    df.drop(df.columns[0], axis=1, inplace=True)
    return df


# ----- core functions -----
# key steps of BPPR
def rotate_BPPR(point_features_str, model, time):
    model_dir = 'data/PALEOGEOGRAPHY_MODELS'
    model_dict = get_reconstruction_model_dict(MODEL_NAME=model)
    static_polygon_filename = str('%s/%s/%s' % (model_dir, model, model_dict['StaticPolygons']))
    rotation_model = pygplates.RotationModel([str('%s/%s/%s' %
                                                  (model_dir, model, rot_file)) for rot_file in
                                              model_dict['RotationFile']])
    partition_time = 0.

    point_features = []
    for point_feature_str in point_features_str:
        lat, lon, p_index = point_feature_str.split(",")
        # lat, lon, p_index = point_feature_str[1]
        # lat, lon, p_index = point_feature_str
        point_feature = pygplates.Feature()
        point_feature.set_geometry(pygplates.PointOnSphere(float(lat), float(lon)))
        point_feature.set_name(str(p_index))
        point_features.append(point_feature)

    # if reconstruct points from past to the present
    # the partition_time will not be 0
    assigned_point_features = pygplates.partition_into_plates(
        static_polygon_filename,
        rotation_model,
        point_features,
        properties_to_copy=[
            pygplates.PartitionProperty.reconstruction_plate_id,
            pygplates.PartitionProperty.valid_time_period],
        reconstruction_time=partition_time
    )

    assigned_point_feature_collection = pygplates.FeatureCollection(assigned_point_features)
    reconstructed_feature_geometries = []
    pygplates.reconstruct(
        assigned_point_feature_collection,
        rotation_model,
        reconstructed_feature_geometries,
        time,
        anchor_plate_id=0)
    result = []
    for reconstructed_feature_geometry in reconstructed_feature_geometries:
        index = reconstructed_feature_geometry.get_feature().get_name()
        rlat, rlon = reconstructed_feature_geometry.get_reconstructed_geometry().to_lat_lon()
        # plat, plon = reconstructed_feature_geometry.get_present_day_geometry().to_lat_lon()
        result.append([index, rlat, rlon])

    return result


# perform paleogeographic point rotation
def rotate_points(points_file, lon_name, lat_name, time, model, output_file):
    points = read_points(points_file)
    data = []
    for index, row in points.iterrows():
        lat = row[lat_name]
        lon = row[lon_name]
        if lat is None or lon is None or lat == "" or lon == "":
            continue
        data.append(str.join(',', [str(lat), str(lon), str(index)]))

    logging.info("Start reconstruct")
    conf = SparkConf().setAppName("Reconstruct with Spark").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    # define partition number
    pointRDD = sc.parallelize(data, 16)
    # reconstruct data in split
    r_start = datetime.datetime.now()
    reconstructedPointRDD = pointRDD.mapPartitions(lambda x: rotate_BPPR(point_features_str=x, model=model, time=time))
    result = reconstructedPointRDD.collect()
    r_end = datetime.datetime.now()
    rotation_time = (r_end - r_start).total_seconds()
    print('rotation_time:', rotation_time)

    # output
    new_lon = "lon_" + str(time) + "MA"
    new_lat = "lat_" + str(time) + "MA"
    new = pd.DataFrame(result, columns=['index', new_lat, new_lon])
    new['index'] = new['index'].astype('int64')
    new.set_index('index', inplace=True)
    # print(new.index)
    out = points.merge(new, how='left', left_index=True, right_index=True)
    # print(out)
    out.to_csv(output_file)


GRID_CONFIG = {
    'lat_range': (-90.0, 90.0),
    'lon_range': (-180.0, 180.0),
    'lat_step': 20.0,
    'lon_step': 40.0
}


class SpatialPartitioner:
    """二维空间网格分区器"""
    def __init__(self, n, lat_range=(-90, 90), lon_range=(-180, 180)):
        self.n = n
        self.lat_min, self.lat_max = lat_range
        self.lon_min, self.lon_max = lon_range

        # 计算网格步长
        self.lat_step = (self.lat_max - self.lat_min) / n
        self.lon_step = (self.lon_max - self.lon_min) / n
        self.num_partitions = n * n

    def get_partition(self, lat, lon):
        """计算网格位置并转换为唯一分区ID"""
        # 边界约束
        lat = max(min(lat, self.lat_max - 1e-6), self.lat_min)
        lon = max(min(lon, self.lon_max - 1e-6), self.lon_min)

        # 计算网格坐标
        row = int((lat - self.lat_min) // self.lat_step)
        col = int((lon - self.lon_min) // self.lon_step)

        # 生成分区ID
        return row * self.n + col

    def partition(self, rdd):
        """执行分区操作"""
        return rdd.map(lambda x: (self.get_partition(x[0], x[1]), x)) \
            .partitionBy(self.num_partitions)


def rotate_points_spatial(points_file, lon_name, lat_name, time, model, output_file):
    points = read_points(points_file)
    data = []
    for index, row in points.iterrows():
        lat = row[lat_name]
        lon = row[lon_name]
        if lat is None or lon is None or lat == "" or lon == "":
            continue
        data.append([lat, lon, index])
    logging.info("Start reconstruct")
    conf = SparkConf().setAppName("Reconstruct with Spark").setMaster("local[8]")
    sc = SparkContext(conf=conf)

    pointRDD = sc.parallelize(data, 8)

    # repartition
    # partitioner = SpatialPartitioner(n=16)
    # pointRDD = partitioner.partition(pointRDD)

    r_start = datetime.datetime.now()
    # reconstruct data in split
    reconstructedPointRDD = pointRDD.mapPartitions(lambda x: rotate_BPPR(point_features_str=x, model=model, time=time))
    result = reconstructedPointRDD.collect()

    r_end = datetime.datetime.now()
    rotation_time = (r_end - r_start).seconds
    print('rotation_time:', rotation_time)

    # output
    new_lon = "lon_paleo"
    new_lat = "lat_paleo"
    new = pd.DataFrame(result, columns=['index', new_lat, new_lon])
    new['index'] = new['index'].astype('int64')
    new.set_index('index', inplace=True)
    # print(new.index)
    out = points.merge(new, how='left', left_index=True, right_index=True)
    # print(out)
    out.to_csv(output_file)


if __name__ == '__main__':
    # define parameters
    time = 70
    model = 'SCOTESE&WRIGHT2018'
    anchor_plate_id = 0
    lon_name = 'longitude'
    lat_name = 'latitude'
    points_file = '../data/syntheticData/point1000000/point1000000.csv'
    output_file = 'data/points/out_point-0306.csv'

    # start to use BPPR
    start = datetime.datetime.now()
    print("Start reconstruct")
    rotate_points(points_file=points_file, lon_name=lon_name, lat_name=lat_name,
                  time=time, model=model, output_file=output_file)
    end = datetime.datetime.now()
    exetime = (end - start).total_seconds()
    print("total_time:" + str(exetime))
