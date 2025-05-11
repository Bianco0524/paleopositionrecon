import time
import os
import json
import math
import pandas as pd
import numpy as np
import multiprocessing
import geohash2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_absolute_error, mean_squared_error
from shapely import Polygon, wkt

from new_data import get_age_partition, rotation_compute
from zm_index import Node, Array


class ST2Plate_Index:
    def __init__(self, model_path=None):
        self.stages = None
        self.non_leaf_stage_len = 0
        self.max_key = 0
        self.rmi = None
        self.model_path = model_path

        self.weight = None
        self.cores = None
        self.train_step = None
        self.batch_num = None
        self.learning_rate = None

    def build(self, X, y,
              stages, weight, cores, train_steps, batch_nums, learning_rates,
              use_thresholds, thresholds, retrain_time_limits, thread_pool_size):
        self.weight = weight
        self.cores = cores[-1]
        self.train_step = train_steps[-1]
        self.batch_num = batch_nums[-1]
        self.learning_rate = learning_rates[-1]

        model_hdf_dir = os.path.join(self.model_path, "hdf/")
        if os.path.exists(model_hdf_dir) is False:
            os.makedirs(model_hdf_dir)
        model_png_dir = os.path.join(self.model_path, "png/")
        if os.path.exists(model_png_dir) is False:
            os.makedirs(model_png_dir)

        self.stages = stages
        stage_len = len(stages)
        self.non_leaf_stage_len = stage_len - 1
        train_inputs = [[[] for j in range(stages[i])] for i in range(stage_len)]
        train_labels = [[[] for j in range(stages[i])] for i in range(stage_len)]
        self.rmi = [None for i in range(stage_len)]

        data_len = len(X)
        self.max_key = data_len
        train_inputs[0][0] = X
        train_labels[0][0] = y

        for i in range(stage_len):
            core = cores[i]
            train_step = train_steps[i]
            batch_num = batch_nums[i]
            learning_rate = learning_rates[i]
            use_threshold = use_thresholds[i]
            threshold = thresholds[i]
            retrain_time_limit = retrain_time_limits[i]
            pool = multiprocessing.Pool(processes=thread_pool_size)
            task_size = stages[i]
            manager = multiprocessing.Manager()
            mp_list = manager.list([None] * task_size)
            train_input = train_inputs[i]
            train_label = train_labels[i]

            # 1. create non-leaf node
            if i < self.non_leaf_stage_len:
                for j in range(task_size):
                    inputs = train_input[j]
                    labels = train_label[j]
                    pool.apply_async(build_nn, args=(self.model_path, i, j, inputs, labels,
                                                     weight, core, train_step, batch_num, learning_rate,
                                                     use_threshold, threshold, retrain_time_limit, mp_list))
                pool.close()
                pool.join()
                nodes = [Node(None, model, None) for model in mp_list]
                for j in range(task_size):
                    node = nodes[j]
                    if node is None:
                        continue
                    else:
                        for ind in range(len(train_input[j])):
                            pre = node.model.predict(train_input[j][ind][2])
                            train_inputs[i + 1][pre].append(train_input[j][ind])
                            train_labels[i + 1][pre].append(train_label[j][ind])
            else:
                for j in range(task_size):
                    inputs = train_input[j]
                    labels = train_label[j]
                    pool.apply_async(build_nn, args=(self.model_path, i, j, inputs, labels,
                                                     weight, core, train_step, batch_num, learning_rate,
                                                     use_threshold, threshold, retrain_time_limit, mp_list))
                pool.close()
                pool.join()
                nodes = [Node(train_input[j], mp_list[j], Array()) for j in range(task_size)]

            self.rmi[i] = nodes
            train_inputs[i] = None
            train_labels[i] = None

    def get_leaf_node(self, key):
        node_key = 0
        for i in range(0, self.non_leaf_stage_len):
            node_key = int(self.rmi[i][node_key].model.predict(key))
        return node_key

    def predict(self, key):
        node_key = self.get_leaf_node(key)
        leaf_node = self.rmi[-1][node_key]
        if leaf_node.model is None:
            while self.rmi[-1][node_key].model is None:
                node_key -= 1
                if node_key <= 0:
                    break
            return self.rmi[-1][node_key], node_key
        # self.model = tf.keras.model.load_model(self.model_path + "model.h5.keras")
        # y_pred = self.model.predict(key)
        # return y_pred
        pre = int(leaf_node.model.predict(key))
        return leaf_node, node_key, pre

    def query(self, point):
        # compute st_keys
        age = get_age_partition(point[2])
        geo_key = geohash2.encode(point[1], point[0], precision=4)
        st_key_int = code_to_int(age_to_code(age) + geo_key)

        y_pred = self.predict(st_key_int)
        y_pred_flattened = np.array([item[0] for item in y_pred])  # 将二维预测值展平为一维

        print(y_pred_flattened)
        """
        # 计算 MAE 和 MSE
        mae = mean_absolute_error(y_test_flattened, y_pred_flattened)
        mse = mean_squared_error(y_test_flattened, y_pred_flattened)

        # 输出评估结果
        print(f"Test MAE: {mae}")
        print(f"Test MSE: {mse}")
        """


class NN:
    def __init__(self, model_path, model_key,
                 data_x, data_y,
                 weight, core, train_step, batch_size, learning_rate,
                 use_threshold, threshold, retrain_time_limit):
        self.name = "Multi-label Classification"
        self.model_path = model_path
        self.model_key = model_key

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(data_x, data_y,
                                                                                test_size=0.2, random_state=42)
        self.weight = weight
        self.core = core
        self.train_step = train_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.retrain_times = 0
        self.retrain_time_limit = retrain_time_limit

        self.model = None

    def init(self):
        self.model = tf.keras.Sequential()
        for i in range(len(self.core) - 1):
            self.model.add(tf.keras.layers.Dense(units=self.core[i + 1],
                                                 input_dim=self.core[i],
                                                 activation='relu'))
        self.model.add(tf.keras.layers.Dense(units=1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def train(self):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor='loss', verbose=1,
                                                        save_best_only=True, mode='min', save_freq='epoch')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.train_step // 100,
                                                          verbose=1, mode='min', restore_best_weights=True)
        self.model.fit(self.train_x, self.train_y,
                       epochs=self.train_step, initial_epoch=0,
                       batch_size=self.batch_size, verbose=1, callbacks=[checkpoint, early_stopping])

        loss, mae = self.model.evaluate(self.test_x, self.test_y)
        if self.use_threshold:
            if self.retrain_times < self.retrain_time_limit:
                self.retrain_times += 1
                self.train()

        print(f"Test Loss: {loss}, Test MAEL: {mae}")
        self.model.save(self.model_path + "model.h5.keras")

    def build(self):
        self.init()
        self.train()


def build_nn(model_path, curr_stage, current_stage_step, inputs, labels,
             weight, core, train_step, batch_num, learning_rate,
             use_threshold, threshold, retrain_time_limit, mp_list=None):
    batch_size = 2 ** math.ceil(math.log(len(inputs) / batch_num, 2))
    if batch_size < 1:
        batch_size = 1
    model_key = "%s_%s" % (curr_stage, current_stage_step)
    """
    tmp_index = NN(model_path, model_key, inputs, labels, 
                   weight, core, train_step, batch_size, learning_rate, use_threshold, threshold,
                   retrain_time_limit)
    tmp_index.build()
    abstract_index = AbstractNN(tmp_index.get_matrices(), len(core) - 1,
                                int(tmp_index.train_x_min), int(tmp_index.train_x_max),
                                int(tmp_index.train_y_min), int(tmp_index.train_y_max),
                                math.floor(tmp_index.min_err), math.ceil(tmp_index.max_err))
    mp_list[current_stage_step] = abstract_index
    """
    tmp_index = NN(model_path, model_key, inputs, labels,
                   weight, core, train_step, batch_size, learning_rate, use_threshold, threshold,
                   retrain_time_limit)
    tmp_index.build()
    mp_list[current_stage_step] = tmp_index


def split_plate_keys(plate_keys):
    # nan
    if type(plate_keys) == float:
        return 'None'
    # multivalued
    if plate_keys.startswith('['):
        plate_keys = plate_keys[1: -1]
        plate_keys_list = plate_keys.split(',')
        return [key.strip().strip("'") for key in plate_keys_list]
    # unique value
    return plate_keys.strip().strip("'")


def age_to_code(input_age):
    # age存在小数如48.1、79.1，因此需要放大10倍； 对4500的处理--大于540赋为550； 取值范围-[0, 550]
    # 2位base32编码 → 0-1023；放大10倍后需要3位base32编码标识 → 0-32767
    __base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    value = ''
    age = int(input_age * 10)
    while age > 0:
        value = __base32[age % 32] + value
        age //= 32
    # 填充3位
    value = value.zfill(3)
    return value


def code_to_int(code):
    __base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    value = 0
    for str in code:
        value = value * 32 + __base32.index(str)
    return value


def model_build(model_path):
    # 1. load data
    data_path = "../data/tested_grids_" + str(280) + "ma.csv"
    data_df = pd.read_csv(data_path)
    # test_data_path = "../data/test_data.csv"
    # data_df = pd.read_csv(test_data_path)
    data_df['plate_keys'] = data_df['plate_id'].apply(split_plate_keys)

    # 2. process labels and save
    all_plate_keys = set()
    for plates in data_df['plate_keys'].dropna():
        if isinstance(plates, list):
            all_plate_keys.update(plates)
        else:
            all_plate_keys.add(plates)
    plate_encoder = LabelEncoder()
    plate_encoder.fit(list(all_plate_keys))

    def encode_plate_keys(plate_list):
        # if isinstance(plate_list, list):
        #     return [plate_encoder.transform([key])[0] for key in plate_list]
        # else:
        #     return plate_encoder.transform([plate_list])[0]
        if isinstance(plate_list, list):
            return plate_encoder.transform(plate_list).tolist()
        elif isinstance(plate_list, str):
            return [plate_encoder.transform([plate_list])[0]]
        else:
            return []

    data_df['plate_codes'] = data_df['plate_keys'].apply(encode_plate_keys)
    # save the mapping for future use
    plate_mapping = {str(k): int(v) for k, v in zip(plate_encoder.classes_, plate_encoder.transform(plate_encoder.classes_))}

    with open(os.path.join(model_path, 'plate_mapping.json'), 'w') as f:
        json.dump(plate_mapping, f)

    # 3. choose features and labels
    X = data_df[['st_keys_int']].values
    # y = data_df['plate_codes'].values
    mlb = MultiLabelBinarizer(classes=plate_encoder.transform(plate_encoder.classes_))
    y = mlb.fit_transform(data_df['plate_codes'])

    # 4. split train and test dataset
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. crate model
    core = [1, 20, 40, 20, 1]
    learning_rate = 0.001
    batch_num = 32
    train_step = 5000

    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    batch_size = 2 ** math.ceil(math.log(len(X) / batch_num, 2))
    if batch_size < 1:
        batch_size = 1

    model = tf.keras.Sequential()
    for i in range(len(core) - 1):
        model.add(tf.keras.layers.Dense(units=core[i + 1],
                                        input_dim=core[i],
                                        activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))  # 添加 Dropout
    model.add(tf.keras.layers.Dense(units=len(mlb.classes_), activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])  # sparse_categorical_crossentropy
    # 6. train model
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=train_step // 100,
                                                      verbose=1, mode='min', restore_best_weights=True)

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y.flatten())
    class_weight_dict = dict(enumerate(class_weights))

    model.fit(X, y, batch_size=batch_size, epochs=train_step, verbose=1, class_weight=class_weight_dict,
              callbacks=[early_stopping], initial_epoch=0)
    # print(res)
    # 7. evaluate model
    # loss, mae = model.evaluate(X_test, y_test)
    # print(f"Test Loss: {loss}, Test MAEL: {mae}")

    # 8. save model
    model.save(model_path + "model.h5.keras")
    return


def model_build_1225(model_path):
    ages = [270, 280, 300, 360, 420, 440, 540]
    data_df = pd.DataFrame()
    for age in ages:
        name = "../data/region_" + str(age) + "_ma.csv"
        df = pd.read_csv(name)
        data_df = pd.concat([data_df, df], ignore_index=True)
    data_df['plate_keys'] = data_df['plate_keys'].apply(split_plate_keys)

    all_plate_keys = set()
    for plates in data_df['plate_keys'].dropna():
        if isinstance(plates, list):
            all_plate_keys.update(plates)
        else:
            all_plate_keys.add(plates)
    plate_encoder = LabelEncoder()
    plate_encoder.fit(list(all_plate_keys))

    def encode_plate_keys(plate_list):
        if isinstance(plate_list, list):
            return plate_encoder.transform(plate_list).tolist()
        elif isinstance(plate_list, str):
            return [plate_encoder.transform([plate_list])[0]]
        else:
            return []

    data_df['plate_codes'] = data_df['plate_keys'].apply(encode_plate_keys)
    plate_mapping = {str(k): int(v) for k, v in
                     zip(plate_encoder.classes_, plate_encoder.transform(plate_encoder.classes_))}

    with open(os.path.join(model_path, 'plate_mapping3.json'), 'w') as f:
        json.dump(plate_mapping, f)

    data_df['geo_keys_int'] = data_df['geo_keys'].apply(code_to_int).values.astype(np.int32)
    X = data_df[['geo_keys_int', 'age']].values
    mlb = MultiLabelBinarizer(classes=plate_encoder.transform(plate_encoder.classes_))
    y = mlb.fit_transform(data_df['plate_codes'])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))

    model.add(tf.keras.layers.Dense(units=len(mlb.classes_), activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'])  # sparse_categorical_crossentropy
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5000 // 100,
                                                      verbose=1, mode='min', restore_best_weights=True)
    model.fit(X, y, batch_size=64, epochs=5000, verbose=1, callbacks=[early_stopping])
    model.save(model_path + "model3.h5.keras")
    return



def model_build_1224_2(model_path):
    ages = [270, 280, 300, 360, 420, 440, 540]
    data_df = pd.DataFrame()
    for age in ages:
        name = "../data/region_" + str(age) + "_ma.csv"
        df = pd.read_csv(name)
        data_df = pd.concat([data_df, df], ignore_index=True)
    data_df['plate_keys'] = data_df['plate_keys'].apply(split_plate_keys)

    all_plate_keys = set()
    for plates in data_df['plate_keys'].dropna():
        if isinstance(plates, list):
            all_plate_keys.update(plates)
        else:
            all_plate_keys.add(plates)
    plate_encoder = LabelEncoder()
    plate_encoder.fit(list(all_plate_keys))

    def encode_plate_keys(plate_list):
        if isinstance(plate_list, list):
            return plate_encoder.transform(plate_list).tolist()
        elif isinstance(plate_list, str):
            return [plate_encoder.transform([plate_list])[0]]
        else:
            return []

    data_df['plate_codes'] = data_df['plate_keys'].apply(encode_plate_keys)
    plate_mapping = {str(k): int(v) for k, v in
                     zip(plate_encoder.classes_, plate_encoder.transform(plate_encoder.classes_))}

    with open(os.path.join(model_path, 'plate_mapping2.json'), 'w') as f:
        json.dump(plate_mapping, f)

    years = data_df['age'].values.astype(np.int32)
    spatial = data_df['geo_keys'].apply(code_to_int).values.astype(np.int32)
    mlb = MultiLabelBinarizer(classes=plate_encoder.transform(plate_encoder.classes_))
    y = mlb.fit_transform(data_df['plate_codes'])
    year_max = years.max()
    spatial_max = spatial.max()

    years_input = tf.keras.layers.Input(shape=(1,), name='year_input')
    spatial_input = tf.keras.layers.Input(shape=(1,), name='spatial_input')
    year_embedding = tf.keras.layers.Embedding(input_dim=year_max + 1, output_dim=16)(years_input)
    year_embedding = tf.keras.layers.Flatten()(year_embedding)
    spatial_embedding = tf.keras.layers.Embedding(input_dim=spatial_max + 1, output_dim=16)(spatial_input)
    spatial_lstm = tf.keras.layers.LSTM(64)(spatial_embedding)
    merged = tf.keras.layers.concatenate([year_embedding, spatial_lstm])
    output = tf.keras.layers.Dense(y.shape[1], activation='sigmoid')(merged)

    model = tf.keras.models.Model(inputs=[years_input, spatial_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    X = np.column_stack((years, spatial))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit([X_train[:, 0], X_train[:, 1]], y_train, batch_size=16, epochs=50, verbose=1,
              validation_data=([X_test[:, 0], X_test[:, 1]], y_test))
    # y_pred = model.predict([X_test[:, 0], X_test[:, 1]])
    # y_pred_class = np.round(y_pred).astype(int)
    # reverse_plate_mapping = {int(v): k for k, v in plate_mapping.items()}
    # predicted_plate_keys = reverse_plate_mapping[y_pred_class]
    test_loss, test_acc = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test, verbose=2)
    print(f"Test Loss: {test_loss}, Test accuracy: {test_acc}")
    model.save(model_path + "model2.h5.keras")
    return


def model_build_1224(model_path):
    ages = [255, 280, 305, 306, 363, 404, 420, 425, 458, 541]
    data_df = pd.DataFrame()
    for age in ages:
        name = "../data/region/region_" + str(age) + "_ma.csv"
        df = pd.read_csv(name)
        data_df = pd.concat([data_df, df], ignore_index=True)
    data_df['plate_keys'] = data_df['plate_keys'].apply(split_plate_keys)

    all_plate_keys = set()
    for plates in data_df['plate_keys'].dropna():
        if isinstance(plates, list):
            all_plate_keys.update(plates)
        else:
            all_plate_keys.add(plates)
    plate_encoder = LabelEncoder()
    plate_encoder.fit(list(all_plate_keys))

    def encode_plate_keys(plate_list):
        if isinstance(plate_list, list):
            return plate_encoder.transform(plate_list).tolist()
        elif isinstance(plate_list, str):
            return [plate_encoder.transform([plate_list])[0]]
        else:
            return []

    data_df['plate_codes'] = data_df['plate_keys'].apply(encode_plate_keys)
    plate_mapping = {str(k): int(v) for k, v in
                     zip(plate_encoder.classes_, plate_encoder.transform(plate_encoder.classes_))}

    with open(os.path.join(model_path, 'plate_mapping.json'), 'w') as f:
        json.dump(plate_mapping, f)

    X = data_df[['longitude', 'latitude', 'age']].values
    mlb = MultiLabelBinarizer(classes=plate_encoder.transform(plate_encoder.classes_))
    y = mlb.fit_transform(data_df['plate_codes'])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))

    model.add(tf.keras.layers.Dense(units=len(mlb.classes_), activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'])  # sparse_categorical_crossentropy
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5000 // 100,
                                                      verbose=1, mode='min', restore_best_weights=True)
    model.fit(X, y, batch_size=64, epochs=5000, verbose=1, callbacks=[early_stopping])
    model.save(model_path + "model.h5.keras")
    return


def model_test(model_path):
    model = tf.keras.models.load_model(model_path + "model.h5.keras")
    with open(os.path.join(model_path, 'plate_mapping.json'), 'r') as f:
        plate_mapping = json.load(f)
    reverse_plate_mapping = {int(v): k for k, v in plate_mapping.items()}

    # X_test = np.array([[38.5, 59.49, 270], [20.38, 50.83, 420], [24.65, 59.30, 540]])  # 301 305 301
    data_path = "../data/region_real.csv"
    data_df = pd.read_csv(data_path)
    data_df['age'] = data_df['age'].apply(get_age_partition)
    X_test = data_df[['lng', 'lat', 'age']].values
    """
    # model3
    spatial = []
    for index, row in data_df.iterrows():
        geo = geohash2.encode(row[5], row[4], precision=4)
        spatial.append(code_to_int(geo))
    # X_test = np.column_stack((data_df['age'].values, np.asarray(spatial)))
    data_df['geo_keys_int'] = spatial
    X_test = data_df[['geo_keys_int', 'age']].values
    """
    y_real = data_df['points_plate_id'].values
    # st_keys_array = np.array(data_df[['st_keys_int']]).reshape(-1, 1)
    # pres = model.predict([X_test[:, 0], X_test[:, 1]])
    pres = model.predict(X_test)
    results = []
    for pre in pres:
        highest_prob_index = np.argmax(pre)
        highest_prob_label = reverse_plate_mapping[highest_prob_index]
        if highest_prob_label == 'None':
            results.append('None')
        else:
            # print(highest_prob_label)
            results.append(highest_prob_label[-3:])
    # print(results)
    res_df = pd.DataFrame()
    res_df['pres'] = results
    res_df['real'] = y_real
    count = 0
    for index, row in res_df.iterrows():
        if row['pres'] != str(row['real']):
            count += 1
    print(count)
    # print(res_df)


def point_predict(model, reverse_plate_mapping, rot, point, age):
    X_test = np.array([[point[0], point[1], age]])
    pres = model.predict(X_test)
    highest_prob_index = np.argmax(pres)
    highest_prob_label = reverse_plate_mapping[highest_prob_index]
    print(highest_prob_label)
    if highest_prob_label == 'None':
        return None
    else:
        plate_hierarchy = highest_prob_label.lstrip('0')
        plates = [int(plate_hierarchy[i:i + 3]) for i in range(0, len(plate_hierarchy), 3)]
        r1_steps = []
        r2_steps = []
        r1_rotations = []
        r2_rotations = []
        for plate in plates:
            # 获取rotation信息
            tmp_df = rot.query('plate_id == @plate and r2_step >= @age and r1_step <= @age')
            if len(tmp_df) == 1:
                r1_steps.append(tmp_df.iloc[0, 2])
                r2_steps.append(tmp_df.iloc[0, 3])
                r1_rotations.append(tmp_df.iloc[0, 4])
                r2_rotations.append(tmp_df.iloc[0, 5])
            else:
                print('Error')
                return
        # print(r1_rotations)
        res = rotation_compute(r1_steps, r2_steps, r1_rotations, r2_rotations, point[0], point[1], age)
    return res


def point_scene(model_path, rot_path):
    rot_df = pd.read_csv(rot_path)
    model = tf.keras.models.load_model(model_path + "model.h5.keras")
    with open(os.path.join(model_path, 'plate_mapping.json'), 'r') as f:
        plate_mapping = json.load(f)
    reverse_plate_mapping = {int(v): k for k, v in plate_mapping.items()}
    # data
    point = [38.5, 59.49]
    age = 253.021
    res = point_predict(model, reverse_plate_mapping, rot_df, point, age)
    return res


def polygon_predict(model, rot_df, reverse_plate_mapping, polygon_wkt, age):
    # 测试polygon
    polygon = wkt.loads(polygon_wkt)
    vertex_array = np.asarray(polygon.exterior.coords)
    points = []
    for vertex in vertex_array:
        point = point_predict(model, reverse_plate_mapping, rot_df, vertex, age)
        points.append(point)
    new_polygon = Polygon(points)
    return new_polygon.wkt


def raster_scene(model_path, rot_path):
    rot_df = pd.read_csv(rot_path)
    model = tf.keras.models.load_model(model_path + "model.h5.keras")
    with open(os.path.join(model_path, 'plate_mapping.json'), 'r') as f:
        plate_mapping = json.load(f)
    reverse_plate_mapping = {int(v): k for k, v in plate_mapping.items()}
    age = 253
    polygon_wkt = "POLYGON((24.770807 54.605282,28.040062 55.837823,30.699796 55.273702,33.137884 53.499383,30.42274 53.268033,28.206296 52.499005,27.319718 53.367337,25.158685 53.499383,24.327518 54.185913,24.770807 54.605282))"
    res = polygon_predict(model, rot_df, reverse_plate_mapping, polygon_wkt, age)
    print(res)


if __name__ == '__main__':
    model_path = "../model_test/"
    rot_path = "../paleo-Rotation-data/PALEOMAP_rot_pairs.csv"
    # model_build(model_path)
    # model_build_1224(model_path)
    # model_build_1224_2(model_path)
    # model_build_1225(model_path)
    # model_test(model_path)
    point_scene(model_path, rot_path)
    # raster_scene(model_path, rot_path)
