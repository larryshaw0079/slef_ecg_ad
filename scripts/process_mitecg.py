import argparse
import os
import warnings

import numpy as np
import scipy.io as sio
import wfdb
from biosppy.signals import ecg
from tqdm.std import tqdm

# Patient candidates. Exclude [102, 104, 107, 217] due to poor quality
PATIENTS = [100, 101, 103, 105, 106, 108, 109,
            111, 112, 113, 114, 115, 116, 117, 118, 119,
            121, 122, 123, 124,
            200, 201, 202, 203, 205, 207, 208, 209,
            210, 212, 213, 214, 215, 219,
            220, 221, 222, 223, 228, 230, 231, 232, 233, 234]
N = {"N", "L", "R"}
S = {"a", "J", "A", "S", "j", "e"}
V = {"V", "E"}
F = {"F"}
Q = {"/", "f", "Q"}

# All beats
BEATS = N.union(S, V, F, Q)
# Anomalous beats
ANOMALOUS_BEATS = S.union(V, F, Q)


def parse_args():
    arg_parser = argparse.ArgumentParser(description='data preprocessing')
    arg_parser.add_argument('--data', dest='data_dir', type=str, required=True)
    arg_parser.add_argument('--dest', dest='dest_dir', type=str, required=True)

    arg_parser.add_argument('--left', dest='left_range', type=int, default=140)
    arg_parser.add_argument('--right', dest='right_range', type=int, default=180)

    return arg_parser.parse_args()


def process_patient(patient_id: int, data_dir: str, left_range: int, right_range: int):
    samples = []
    labels = []

    # Read record and annotation
    record = wfdb.rdrecord(os.path.join(data_dir) + '/' + str(patient_id))
    annotation = wfdb.rdann(os.path.join(data_dir) + '/' + str(patient_id), 'atr')

    # Flip the channel for patient 114
    if patient_id != 114:
        signal1 = record.p_signal[:, 0].reshape(-1)
        signal2 = record.p_signal[:, 1].reshape(-1)
    else:
        signal1 = record.p_signal[:, 1].reshape(-1)
        signal2 = record.p_signal[:, 0].reshape(-1)

    # Smooth signals
    sig_out1 = ecg.ecg(signal=signal1, sampling_rate=record.fs, show=False)
    signal1 = sig_out1['filtered']

    sig_out2 = ecg.ecg(signal=signal2, sampling_rate=record.fs, show=False)
    signal2 = sig_out2['filtered']

    # Reading r-peaks
    r_peaks = sig_out1['rpeaks']

    # Reading annotations. `symbol` and `sample` are labels and values respectively.
    ann_symbol = annotation.symbol
    ann_sample = annotation.sample

    # Iterate annotations
    for idx, symbol in enumerate(ann_symbol):
        if symbol in BEATS:
            ann_idx = ann_sample[idx]
            if ann_idx - left_range >= 0 and ann_idx + right_range < record.sig_len:
                if symbol in N:
                    closest_r_peak = r_peaks[np.argmin(np.abs(r_peaks - ann_idx))]
                    if abs(closest_r_peak - ann_idx) < 10:
                        # samples.append(([signal1[ann_idx - left_range:ann_idx + right_range],
                        #                  signal2[ann_idx - left_range:ann_idx + right_range]], 'N', symbol))
                        samples.append([signal1[ann_idx - left_range:ann_idx + right_range],
                                        signal2[ann_idx - left_range:ann_idx + right_range]])
                        # labels.append(('N', symbol))
                        labels.append('N')
                else:
                    aami_label = ''
                    if symbol in S:
                        aami_label = 'S'
                    elif symbol in V:
                        aami_label = 'V'
                    elif symbol in F:
                        aami_label = 'F'
                    elif symbol in Q:
                        aami_label = 'Q'
                    else:
                        raise ValueError('Invalid annotation type!')

                    # samples.append(([signal1[ann_idx - left_range:ann_idx + right_range], signal2[ann_idx - left_range:ann_idx + right_range]],
                    #                 aami_label, symbol))
                    samples.append([signal1[ann_idx - left_range:ann_idx + right_range],
                                    signal2[ann_idx - left_range:ann_idx + right_range]])
                    # labels.append((aami_label, symbol))
                    labels.append(aami_label)

    return np.asarray(samples), np.asarray(labels)


def prepare_data(data_dir: str, dest_dir: str, left_range: int, right_range: int):
    print('====================BEGINNING====================')

    data_x = []
    data_y = []

    for patient_id in tqdm(PATIENTS):
        # samples: (num_sample, num_channel, length), labels: (num_sample)
        samples, labels = process_patient(patient_id, data_dir, left_range, right_range)

        data_x.append(samples)
        data_y.append(labels.reshape((-1, 1)))

    data_x = np.concatenate(data_x)
    data_y = np.concatenate(data_y)

    # if anomaly_ratio > 0.0:
    #     anomaly_num = V_samples.shape[0] + S_samples.shape[0] + F_samples.shape[0] + Q_samples.shape[0]
    #     normal_num = N_samples.shape[0]
    #     N_num = N_samples.shape[0]
    #     V_num = V_samples.shape[0]
    #     S_num = S_samples.shape[0]
    #     F_num = F_samples.shape[0]
    #     Q_num = Q_samples.shape[0]
    #
    #     new_anomaly_num = int(anomaly_ratio / (1 + anomaly_ratio) * normal_num)
    #     new_V_num = int(new_anomaly_num * V_num / anomaly_num)
    #     new_S_num = int(new_anomaly_num * S_num / anomaly_num)
    #     new_F_num = int(new_anomaly_num * F_num / anomaly_num)
    #     new_Q_num = int(new_anomaly_num * Q_num / anomaly_num)
    #     if new_Q_num == 0:
    #         new_Q_num = 1
    #
    #     N_samples = np.concatenate([N_samples, V_samples[-new_V_num:]])
    #     V_samples = V_samples[:-new_V_num]
    #
    #     N_samples = np.concatenate([N_samples, S_samples[-new_S_num:]])
    #     S_samples = S_samples[:-new_S_num]
    #
    #     N_samples = np.concatenate([N_samples, F_samples[-new_F_num:]])
    #     F_samples = F_samples[:-new_F_num]
    #
    #     N_samples = np.concatenate([N_samples, Q_samples[-new_Q_num:]])
    #     Q_samples = Q_samples[:-new_Q_num]

    if not os.path.exists(dest_dir):
        warnings.warn('Path {} dose not exist, created'.format(dest_dir))
        os.makedirs(dest_dir)

    sio.savemat(os.path.join(dest_dir, 'mit_ecg.mat'), {'data': data_x, 'label': data_y})

    print('====================FINISHED====================')


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # wfdb.dl_database('mitdb', args.data_dir)

    prepare_data(args.data_dir, args.dest_dir, args.left_range, args.right_range)
