from collections import Counter
import pandas as pd
# import vaex
import numpy as np
import glob
import dask.dataframe as dd
import json
import math
import csv
import time
import _warnings
import tensorflow as tf
from tqdm import tqdm
from scipy.stats import entropy
import argparse
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

can_ml_attributes = ['timestamp', 'arbitration_id', 'data_field', 'attack']

def pad_or_truncate(sequence, target_length):
    """Pad the sequence with zeros or truncate to the target length."""
    if len(sequence) < target_length:
        sequence = sequence + [0] * (target_length - len(sequence))  # pad with zeros
    return sequence[:target_length]  # truncate if longer

def reshape_to_target_size(array, target_size=64):
    """Reshape or pad a flattened array into a target_size x target_size matrix."""
    padded = pad_or_truncate(array, target_size * target_size)  # Ensure target_size^2 elements
    return np.array(padded).reshape((target_size, target_size))  # Reshape into target_size x target_size

def split_into_list(string, type):
    res = []
    if type == 'payload':
        res = list(string)

    elif type == 'canID':
        res = list(bin(int(string))[2:].zfill(64))
    
    return [int(bit) for bit in res]

def parse_and_stack_timestamps(timestamps, target_size=64):
    """
    Parse and stack timestamps into a 2D array where each row is a parsed timestamp.
    """
    def parse_single_timestamp(timestamp):
        # Normalize timestamp and convert to binary array
        normalized = (timestamp - np.floor(timestamp)) * (2 ** 32)
        parsed = np.array(list(bin(int(normalized))[2:].zfill(64))).astype(float)
        return parsed

    # Parse and stack each timestamp
    stacked_timestamps = np.array([parse_single_timestamp(ts) for ts in timestamps])

    # Ensure the number of rows matches target_size by truncating or padding
    if stacked_timestamps.shape[0] > target_size:
        stacked_timestamps = stacked_timestamps[:target_size, :]
    elif stacked_timestamps.shape[0] < target_size:
        padding = np.zeros((target_size - stacked_timestamps.shape[0], stacked_timestamps.shape[1]))
        stacked_timestamps = np.vstack((stacked_timestamps, padding))

    return stacked_timestamps

def normalize_time_zscore(time_series):
    """Normalize time using Z-score normalization."""
    mean_val = time_series.mean()
    std_val = time_series.std()
    normalized = (time_series - mean_val) / std_val
    return normalized

# def serialize_example(x, y): 
#     """converts x, y to tf.train.Example and serialize"""
#     id_seq, data_seq, timestamp = x
#     id_seq = tf.train.Int64List(value = np.array(id_seq).flatten())
#     data_seq = tf.train.Int64List(value = np.array(data_seq).flatten())
#     timestamp = tf.train.FloatList(value = np.array(timestamp).flatten())

#     label = tf.train.Int64List(value = np.array([y]))

#     features = tf.train.Features(
#         feature = {
#             "id_seq": tf.train.Feature(int64_list = id_seq),
#             "data_seq": tf.train.Feature(int64_list = data_seq),
#             "timestamp": tf.train.Feature(float_list = timestamp),
#             "label" : tf.train.Feature(int64_list = label)
#         }
#     )
#     example = tf.train.Example(features = features)
#     return example.SerializeToString()

def serialize_example(x, y): 
    id_seq, data_seq, timestamp, delta_ts, id_freq = x
    id_seq = tf.train.Int64List(value=np.array(id_seq).flatten())
    data_seq = tf.train.Int64List(value=np.array(data_seq).flatten())
    timestamp = tf.train.FloatList(value=np.array(timestamp).flatten())
    delta_ts = tf.train.FloatList(value=np.array(delta_ts).flatten())
    id_freq = tf.train.FloatList(value=np.array(id_freq).flatten())

    label = tf.train.Int64List(value=np.array([y]))

    features = tf.train.Features(
        feature={
            "id_seq": tf.train.Feature(int64_list=id_seq),
            "data_seq": tf.train.Feature(int64_list=data_seq),
            "timestamp": tf.train.Feature(float_list=timestamp),
            "delta_ts": tf.train.Feature(float_list=delta_ts),
            "id_freq": tf.train.Feature(float_list=id_freq),
            "label": tf.train.Feature(int64_list=label)
        }
    )
    example = tf.train.Example(features=features)
    return example.SerializeToString()



# def write_tfrecord(data, filename):
#     tfrecord_writer = tf.io.TFRecordWriter(filename)
#     for _, row in tqdm(data.iterrows()):
#         X = (row['id_seq'], row['data_seq'], row['timestamp'])
#         Y = row['label']
#         tfrecord_writer.write(serialize_example(X, Y))
#     tfrecord_writer.close() 

def write_tfrecord(data, filename):
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows()):
        X = (row['id_seq'], row['data_seq'], row['timestamp'], row['delta_ts'], row['id_freq'])
        Y = row['label']
        tfrecord_writer.write(serialize_example(X, Y))
    tfrecord_writer.close()

def split_data(file_name, attack_id, window_size, strided_size, target_size=64):
    if not os.path.exists(file_name):
        print(file_name, ' does not exist!')
        return None

    print("Window size = {}, strided = {}".format(window_size, strided_size))
    df = pd.read_csv(file_name)
    df.columns = can_ml_attributes
    print("Reading {}: done".format(file_name))
    df = df.sort_values('timestamp', ascending=True)

    # Binary 'can data' (CAN Data)
    df['Data'] = df['data_field'].apply(lambda x: split_into_list(x, 'payload'))
    # Binary 'canID' (CAN ID)
    df['canID'] = df['arbitration_id'].apply(lambda x: split_into_list(x, 'canID'))

    # Normalize and stack timestamp
    df['timestamp'] = normalize_time_zscore(df['timestamp'].values)

    print("CAN-ML pre-processing: Done")

    print("Initial attack count before windowing:", df[df['attack'] != 0].shape[0])
    as_strided = np.lib.stride_tricks.sliding_window_view

    canid = as_strided(df.canID.values, window_shape=window_size)[::strided_size]
    data = as_strided(df.Data.values, window_shape=window_size)[::strided_size]
    timestamp = as_strided(df.timestamp.values, window_shape=window_size)[::strided_size]

    label = as_strided(df.attack.values, window_shape=window_size)[::strided_size]
    
  # === Compute raw delta_ts ===
    ts_windows = as_strided(df['timestamp'].values, window_shape=window_size)[::strided_size]

    delta_ts_all = []
    for ts in ts_windows:
        delta = np.diff(ts, prepend=ts[0])  # [0.0, ts[1]-ts[0], ..., ts[n]-ts[n-1]]
        delta_ts_all.append(delta)
    delta_ts_all = np.array(delta_ts_all)  # shape: [N, window_size]

    # === log1p + RobustScaler (reduce effect of large outliers) ===
    log_delta = np.log1p(delta_ts_all)  # more stable for small values
    scaler_dt = RobustScaler(quantile_range=(5.0, 95.0))
    log_delta_scaled = scaler_dt.fit_transform(log_delta)  # shape: [N, window_size]

    # === Clip for numerical stability ===
    log_delta_scaled = np.clip(log_delta_scaled, -3.0, 3.0)

    # === Rescale to [0, 1] for compatibility with Normalize(mean=0.5, std=0.5) ===
    min_dt = log_delta_scaled.min()
    max_dt = log_delta_scaled.max()
    log_delta_scaled = (log_delta_scaled - min_dt) / (max_dt - min_dt + 1e-8)

    # === Expand to [target_size x target_size] ===
    delta_ts = np.array([
        np.tile(row, (target_size, 1))
        for row in log_delta_scaled
    ])  # shape: [N, target_size, target_size]

    # -----------------------------------------------------------------------------

    # === Compute id_freq ===
    canid_windows = as_strided(df['arbitration_id'].values, window_shape=window_size)[::strided_size]
    id_freq_all = []

    for window in canid_windows:
        count_map = Counter(window)
        freqs = [count_map[can_id] for can_id in window]
        id_freq_all.append(freqs)

    id_freq_all = np.array(id_freq_all)  # shape: [N, window_size]

    # === Normalize with StandardScaler ===
    scaler_idf = StandardScaler()
    id_freq_scaled = scaler_idf.fit_transform(id_freq_all)

    # === Clip for stability (optional but recommended) ===
    id_freq_scaled = np.clip(id_freq_scaled, -3.0, 3.0)

    # === Rescale to [0, 1] ===
    min_idf = id_freq_scaled.min()
    max_idf = id_freq_scaled.max()
    id_freq_scaled = (id_freq_scaled - min_idf) / (max_idf - min_idf + 1e-8)

    # === Expand to [target_size x target_size] ===
    id_freq = np.array([
        np.tile(row, (target_size, 1))
        for row in id_freq_scaled
    ])  # shape: [N, target_size, target_size]

    df = pd.DataFrame({
        'id_seq': pd.Series(canid.tolist()),
        'data_seq': pd.Series(data.tolist()),
        'timestamp': pd.Series(timestamp.tolist()),
        'delta_ts': pd.Series(delta_ts.tolist()),
        'id_freq': pd.Series(id_freq.tolist()),
        'label': pd.Series(label.tolist())
    }, index=range(len(canid)))

    df['label'] = df['label'].apply(lambda x: attack_id if any(x) else 0)

    print("Aggregating data: Done")
    print('#Normal: ', df[df['label'] == 0].shape[0])
    print('#Attack: ', df[df['label'] != 0].shape[0])

    df['id_seq'] = df['id_seq'].apply(lambda x: reshape_to_target_size([item for sublist in x for item in sublist], target_size))
    df['data_seq'] = df['data_seq'].apply(lambda x: reshape_to_target_size([item for sublist in x for item in sublist], target_size))
    df['timestamp'] = df['timestamp'].apply(lambda x: parse_and_stack_timestamps(x, target_size))

    return df[['id_seq', 'data_seq', 'timestamp', 'delta_ts', 'id_freq', 'label']].reset_index().drop(['index'], axis=1)



def main(indir, outdir, attacks, window_size, strided, target_size=64):
    print(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data_info = {}

    for attack_id, attack in enumerate(attacks):
        print('Attack: {} ==============='.format(attack))
        finput = '{}/{}.csv'.format(indir, attack)
        print("Attack Id: ", attack_id)
        df = split_data(finput, attack_id + 1, window_size, strided, target_size=target_size)
        print("Writing...................")

        foutput_attack = '{}/{}'.format(outdir, attack)
        foutput_normal = '{}/Normal_{}'.format(outdir, attack)
        df_attack = df[df['label'] != 0].sample(frac=0.05, random_state=32)
        df_normal = df[df['label'] == 0].sample(frac=0.05, random_state=32)
        write_tfrecord(df_attack, foutput_attack)
        write_tfrecord(df_normal, foutput_normal)
        
        data_info[foutput_attack] = df_attack.shape[0]
        data_info[foutput_normal] = df_normal.shape[0]
    
    json.dump(data_info, open('{}/datainfo.txt'.format(outdir), 'w'))
    print("DONE!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--indir', type=str, default="./data/set_04/train_01/merged_no_accessory")
    parser.add_argument('--indir', type=str, default="./data/set_01/train_01/merged")
    parser.add_argument('--outdir', type=str, default="./data/set_01/train_01/preprocessed/six_features/TFRecord")
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--strided', type=int, default=32)
    args = parser.parse_args()
    
    # attack_types = ["double", "fuzzing", "interval", "speed", "systematic", "triple"]
    # attack_types = ["interval", "rpm", "speed", "standstill", "systematic"]
    attack_types = ['dos', 'double', 'force-neutral', 'fuzzing', 'triple']
    # attack_types = ['dos', 'force-neutral', 'rpm', 'standstill']
    # attack_types = ['triple']
    if args.strided is None:
        args.strided = args.window_size
        
    outdir =  args.outdir + '_w{}_s{}'.format(args.window_size, args.strided)
    main(args.indir, outdir, attack_types, args.window_size, args.strided)