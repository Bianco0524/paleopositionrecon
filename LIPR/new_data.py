import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely import wkt
from shapely.ops import unary_union
import fiona
import geohash2
import quaternion as Q

from plate_tree import tree_252_255, tree_255_280, tree_280_305, tree_305_306, tree_306_363, \
    tree_363_404, tree_404_420, tree_420_425, tree_425_458, tree_458_541


def vector(*args):
    return np.array(args)


def unit_vector(*args):
    v = vector(*args)
    return v / np.linalg.norm(v)


def cart2sph(unit_vec):
    (x, y, z) = unit_vec
    lat = np.arcsin(z)
    lon = np.arctan2(y, x)
    return np.degrees(lon), np.degrees(lat)


def sph2cart(lon, lat):
    _lon = np.radians(lon)
    _lat = np.radians(lat)
    x = np.cos(_lat) * np.cos(_lon)
    y = np.cos(_lat) * np.sin(_lon)
    z = np.sin(_lat)
    return unit_vector(x, y, z)


def euler_to_quaternion_2(euler_pole):
    lat, lon, angle = euler_pole.split(",")
    lat = float(lat)
    lon = float(lon)
    angle = float(angle)
    angle = np.radians(angle)
    w = np.cos(angle / 2)
    v = sph2cart(lon, lat) * np.sin(angle / 2)
    return np.quaternion(w, *v)


def get_age_partition(age):
    age_nodes = [255, 280, 305, 306, 363, 404, 420, 425, 458, 541]
    # 对于age“向上取整”作为基准判断板块
    left, right = 0, len(age_nodes) - 1
    while left <= right:
        mid = (left + right) // 2
        if age_nodes[mid] < age:
            left = mid + 1
        else:
            right = mid - 1

    if left < len(age_nodes):
        return age_nodes[left]
    else:
        print("Error: age beyond the range!")
        return


def get_plates_key(plate_tree, plate_id, target_plate_id, path=None, visited=None):
    # 按序拼接plate_id并返回 例如，对plate_id216返回101262205216
    if path is None:
        path = []
    if visited is None:
        visited = set()  # 用于记录已经访问的 plate_id，以防重复访问

    # 如果当前 plate 已经访问过，跳过
    if plate_id in visited:
        return None

    # plate_id=1时，以‘001’记录
    if plate_id == 1:
        path.append('001')
    else:
        path.append(str(plate_id))
    # 标记已访问
    visited.add(plate_id)

    # 找到目标，完成拼接
    if plate_id == target_plate_id:
        return ''.join(path)

    if plate_id in plate_tree:
        for next_plate in plate_tree[plate_id]:
            result = get_plates_key(plate_tree, next_plate, target_plate_id, path, visited)
            if result:
                return result

    # 如果没有找到路径，回溯时移除当前 plate
    path.pop()
    return None


def complete_plates_key(plate_key, target_len=45):
    # 如果plates_key长度不足45位（最长），前面用0补足
    current_len = len(plate_key)
    if current_len < target_len:
        plate_key = plate_key.rjust(target_len, '0')
    return plate_key


def get_unified_plates_key(target_plate_id, age):
    # decide plate_tree according to age
    ideal_age = get_age_partition(age)
    if ideal_age == 255:
        plate_tree = tree_252_255
    elif ideal_age == 280:
        plate_tree = tree_255_280
    elif ideal_age == 305:
        plate_tree = tree_280_305
    elif ideal_age == 306:
        plate_tree = tree_305_306
    elif ideal_age == 363:
        plate_tree = tree_306_363
    elif ideal_age == 404:
        plate_tree = tree_363_404
    elif ideal_age == 420:
        plate_tree = tree_404_420
    elif ideal_age == 425:
        plate_tree = tree_420_425
    elif ideal_age == 458:
        plate_tree = tree_425_458
    elif ideal_age == 541:
        plate_tree = tree_458_541
    else:
        print("age error!")
        return
    res_plate = get_plates_key(plate_tree, 0, target_plate_id)[1:]
    return complete_plates_key(res_plate)


def summation_geometries():
    plates = "../data/plate_geometries.csv"
    plates_df = pd.read_csv(plates)
    old_lims = [255, 280, 305, 306, 363, 404, 420, 425, 458, 541]
    plates_df['geometry'] = plates_df['wkt'].apply(wkt.loads)
    plates_gdf = gpd.GeoDataFrame(plates_df)
    sum_gdf = gpd.GeoDataFrame(columns=['age', 'plate_id', 'geometry'])

    for age in old_lims:
        filter_gdf = plates_gdf[plates_gdf['old_lim'] >= age].copy()
        filter_gdf['age'] = [age] * len(filter_gdf)
        # filter_gdf['geometry'] = filter_gdf['geometry'].apply(convert_multipolygon_to_polygon)
        sum_gdf = pd.concat([sum_gdf, filter_gdf])
    # print(sum_gdf)
    return sum_gdf


def data_process_1230():
    lon_min, lon_max = (20, 40)
    lat_min, lat_max = (50, 60)
    ages = [255, 280, 305, 306, 363, 404, 420, 425, 458, 541]
    reso = 0.1
    lons = np.arange(lon_min, lon_max, reso)
    lats = np.arange(lat_min, lat_max, reso)
    plates_gdf = summation_geometries()
    for age in ages:
        if age != 541:
            continue
        name = "../data/region/region_" + str(age) + "_ma.csv"
        polygons = []
        geo_keys = []
        longitude = []
        latitude = []
        # grids creation
        for i in range(len(lons) - 1):
            for j in range(len(lats) - 1):
                min_x = lons[i]
                max_x = lons[i + 1]
                min_y = lats[j]
                max_y = lats[j + 1]
                polygon = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])
                polygons.append(polygon)
                geohash = geohash2.encode(np.mean([min_y, max_y]), np.mean([min_x, max_x]), precision=4)
                geo_keys.append(geohash)
                longitude.append(np.mean([min_x, max_x]))
                latitude.append(np.mean([min_y, max_y]))
        grids = gpd.GeoDataFrame(columns=['geo_keys', 'longitude', 'latitude', 'age', 'plate_keys'], geometry=polygons)
        grids['geo_keys'] = geo_keys
        grids['longitude'] = longitude
        grids['latitude'] = latitude
        grids['age'] = age

        # plate_id computation
        plate_keys = []
        age_plate_gdf = plates_gdf[plates_gdf['age'] == age].copy()
        for index, row in grids.iterrows():
            age_plate_gdf['is_intersected'] = age_plate_gdf['geometry'].intersects(row['geometry'])
            intersecting_gdf = age_plate_gdf[age_plate_gdf['is_intersected']]
            if len(intersecting_gdf) == 1:
                plate_key = get_unified_plates_key(intersecting_gdf['plate_id'].values, age)
            elif len(intersecting_gdf) == 0:
                plate_key = None
            else:
                plate_key = []
                for plate_id in intersecting_gdf['plate_id'].values:
                    plate_key.append(get_unified_plates_key(plate_id, age))
            plate_keys.append(plate_key)
        grids['plate_keys'] = plate_keys
        grids.to_csv(name)
    return


def rotation_compute(r1_steps, r2_steps, r1_rotations, r2_rotations, lng, lat, age):
    # 假设数据均已处理完毕
    """
    # 示例数据
    age = 253.021
    lng = 38.5
    lat = 59.49
    plate_ids = [701, 714, 101, 301]
    ref_plate_ids = [0, 701, 714, 101]
    r1_steps = [240, 143.8, 175, 175]  # [175, 175, 143.8, 240]
    r2_steps = [255, 1100, 306, 255]  #[255, 306, 1100, 255]
    r1_rotations = '{0,138,44}|{21.99, 11.62, 5.99}|{66.95, -12.02, 75.55}|{-89.1, -133.9, 39.75}'
    r2_rotations = '{0, 148, 51}|{21.99, 11.62, 5.99}|{61.88, -19.36, 82.11}|{-89.1, -133.9, 39.75}'
    # r1_rotations = ['[0.0, 138.0, 44.0]', '[21.99, 11.62, 5.99]', '[66.95, -12.02, 75.55]', '[-89.1, -133.9, 39.75]']
    """

    base = np.quaternion(1, 0, 0, 0)
    # r1 = r1_rotations.split("|")
    # r2 = r2_rotations.split("|")
    for i in range(0, len(r1_steps)):
        temp_r1 = r1_rotations[i].strip("'").replace("[", "").replace("]", "")
        temp_r2 = r2_rotations[i].strip("'").replace("[", "").replace("]", "")
        if r1_steps[i] == age:
            q1 = euler_to_quaternion_2(temp_r1)
            base = base * q1
        else:
            q1 = euler_to_quaternion_2(temp_r1)
            q2 = euler_to_quaternion_2(temp_r2)
            res = Q.slerp(q1, q2, float(r1_steps[i]), float(r2_steps[i]), float(age))
            base = base * res
    # print(base)
    point = [lng, lat]
    v0 = sph2cart(*point)
    v1 = Q.rotate_vectors(base, v0)
    # print(cart2sph(v1))
    return cart2sph(v1)


def rot_pairs():
    rot_path = "../paleo-Rotation-data/PALEOMAP.csv"
    rot_df = pd.read_csv(rot_path)
    plate_id_index = rot_df['plate_id'].unique()

    # plate = 301(if1) plate=601(if2)
    plate_id = []
    ref_plate_id = []
    r1_step = []
    r2_step = []
    r1_rotation = []
    r2_rotation = []

    def make_pairs(data_df):
        for i in range(0, len(data_df) - 1):
            tmp_plate_id = data_df.iloc[i]['plate_id']
            tmp_ref_plate_id = data_df.iloc[i]['ref_plate_id']
            tmp_r1_step = data_df.iloc[i]['age']
            tmp_r2_step = data_df.iloc[i + 1]['age']
            tmp_r1_rotation = [data_df.iloc[i]['latitude'],
                               data_df.iloc[i]['longitude'],
                               data_df.iloc[i]['angle']]
            tmp_r2_rotation = [data_df.iloc[i + 1]['latitude'],
                               data_df.iloc[i + 1]['longitude'],
                               data_df.iloc[i + 1]['angle']]
            plate_id.append(tmp_plate_id)
            ref_plate_id.append(tmp_ref_plate_id)
            r1_step.append(tmp_r1_step)
            r2_step.append(tmp_r2_step)
            r1_rotation.append(tmp_r1_rotation)
            r2_rotation.append(tmp_r2_rotation)

    for plate in plate_id_index:
        filtered_df = rot_df[rot_df['plate_id'] == plate]
        sorted_df = filtered_df.sort_values(by='age', ascending=True)

        # 判断是否存在多个ref_plate_id
        refs = sorted_df['ref_plate_id'].unique()
        # print(len(ref_num))
        if len(refs) > 1:
            # 特殊处理
            # 获取ref_plate_id & age分段
            segments = []
            current_ref = sorted_df.iloc[0, 1]  # ref_plate_id
            start = sorted_df.iloc[0, 2]  # age
            count = 0
            for index, row in sorted_df.iterrows():
                count += 1
                if count == 0:
                    continue
                else:
                    tmp_ref = row['ref_plate_id']
                    tmp_age = row['age']
                    if tmp_ref != current_ref:
                        segments.append((current_ref, start, sorted_df.iloc[count - 1, 2]))  # age index=2
                        current_ref = tmp_ref
                        start = tmp_age
            segments.append((current_ref, start, sorted_df.iloc[-1, 2]))
            # 获得pairs
            for i in range(0, len(segments)):
                start = segments[i][1]
                end = segments[i][2]
                tmp_ref = segments[i][0]
                part_df = sorted_df.query('age >= @start and age <= @end and ref_plate_id == @tmp_ref')
                make_pairs(part_df)
        else:
            # 普通处理
            make_pairs(sorted_df)


    rotation_pairs = pd.DataFrame()
    rotation_pairs['plate_id'] = plate_id
    rotation_pairs['ref_plate_id'] = ref_plate_id
    rotation_pairs['r1_step'] = r1_step
    rotation_pairs['r2_step'] = r2_step
    rotation_pairs['r1_rotation'] = r1_rotation
    rotation_pairs['r2_rotation'] = r2_rotation
    rotation_pairs.to_csv("../paleo-Rotation-data/PALEOMAP_rot_pairs.csv", index=False)


def test_function():
    # q1 = np.quaternion(0.927184,-0.278387,0.250661,0)
    # q2 = np.quaternion(0.902585,-0.365094,0.228136,0)
    base = np.quaternion(0.906282, -0.353819, 0.231225, 0)
    point = [38.5, 59.49]
    v0 = sph2cart(*point)
    res = Q.rotate_vectors(base, v0)
    print(cart2sph(res))


if __name__ == '__main__':
    # data_process_1230()
    # rotation_compute()
    # rot_pairs()
    test_function()

