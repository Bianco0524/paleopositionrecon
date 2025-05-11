import csv
import pygplates
import logging
import os
import pandas as pd
import datetime
import geopandas as gpd
import fiona
from shapely.geometry import shape, Polygon
from shapely import wkt

from BPPR.rotation_utils import get_reconstruction_model_dict


def reconstruct_polygon():
    polygon_path = "../maps/gadm41_LTU_shp/gadm41_LTU_1.shp"
    model_dir = 'data/PALEOGEOGRAPHY_MODELS'
    model_dict = get_reconstruction_model_dict(MODEL_NAME='SCOTESE&WRIGHT2018')
    static_polygon_filename = str('%s/%s/%s' % (model_dir, 'SCOTESE&WRIGHT2018', model_dict['StaticPolygons']))
    rotation_model = pygplates.RotationModel([str('%s/%s/%s' %
                                              (model_dir, 'SCOTESE&WRIGHT2018', rot_file)) for rot_file in
                                              model_dict['RotationFile']])
    feature_collection = pygplates.FeatureCollection()
    with fiona.open(polygon_path, "r") as src:
        for feature in src:
            shapely_geom = shape(feature["geometry"])
            if shapely_geom.geom_type == "MultiPolygon":
                for polygon in shapely_geom.geoms:  # 遍历每个子多边形
                    exterior_coords = polygon.exterior.coords  # 获取外环坐标
                    coords = [(lon, lat) for lon, lat in exterior_coords]
                    pygp_geom = pygplates.PolygonOnSphere(coords)
                    feat = pygplates.Feature()
                    feat.set_geometry(pygp_geom)
                    feature_collection.add(feat)

            elif shapely_geom.geom_type == "Polygon":
                exterior_coords = shapely_geom.exterior.coords
                coords = [(lon, lat) for lon, lat in exterior_coords]
                pygp_geom = pygplates.PolygonOnSphere(coords)
                feat = pygplates.Feature()
                feat.set_geometry(pygp_geom)
                feature_collection.add(feat)

    assigned_point_features = pygplates.partition_into_plates(
        static_polygon_filename,
        rotation_model,
        feature_collection,
        properties_to_copy=[
            pygplates.PartitionProperty.reconstruction_plate_id,
            pygplates.PartitionProperty.valid_time_period],
        reconstruction_time=0.
    )

    assigned_point_feature_collection = pygplates.FeatureCollection(assigned_point_features)
    reconstructed_feature_geometries = []
    pygplates.reconstruct(
        assigned_point_feature_collection,
        rotation_model,
        reconstructed_feature_geometries,
        253,
        anchor_plate_id=0)
    result = []
    for reconstructed_feature_geometry in reconstructed_feature_geometries:
        index = reconstructed_feature_geometry.get_feature().get_name()
        geom = reconstructed_feature_geometry.get_reconstructed_geometry()
        coordinates = [(point.get_longitude(), point.get_latitude()) for point in geom.to_lat_lon_point_list()]
        shapely_polygon = Polygon(coordinates)
        p_wkt = shapely_polygon.wkt
        result.append(p_wkt)
    # print(len(result))
    gdf = gpd.GeoDataFrame(crs=4326)
    gdf['wkt'] = result
    gdf['geometry'] = gdf['wkt'].apply(wkt.loads)
    gdf.to_file("../maps/gadm41_LTU_shp/res.shp", driver="ESRI Shapefile")


if __name__ == '__main__':
    reconstruct_polygon()
    # print()