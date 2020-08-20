import os

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize(series):
    return (series - np.min(series)) / (np.max(series) - np.min(series))


class ECGDataset(Dataset):
    def __init__(self, data, label=None, is_normalize=True, num_channel=2):
        assert data.ndim == 3
        if label is not None:
            assert data.shape[0] == label.shape[0]
            assert label.ndim == 1 or (label.ndim == 2 and label.shape[1] == 1)

        self.data = data[:, :num_channel, :]
        self.label = label.reshape(-1, 1) if label is not None else None

        if is_normalize:
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    self.data[i][j] = normalize(self.data[i][j])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        x = torch.from_numpy(self.data[item].astype(np.float32))
        if self.label is not None:
            y = torch.from_numpy(self.label[item].astype(np.int))
            return x, y
        else:
            return x


def prepare_data(data_path, train_val_test_split=(5, 2, 3), val_anomaly_ratio=0.5, anomaly_injection_ratio=0.0,
                 verbose=True):
    # with open(os.path.join(data_path, 'data.pkl'), 'rb') as f:
    #     data = pickle.load(f)
    with open(os.path.join(data_path, 'N_samples.npy'), 'rb') as f:
        normal_samples = np.load(f)
    # normal_samples = data['N_samples']

    anomalous_samples = []
    for file_name in ['V_samples.npy', 'S_samples.npy', 'F_samples.npy', 'Q_samples.npy']:
        with open(os.path.join(data_path, file_name), 'rb') as f:
            anomalous_samples.append(np.load(f))
    anomalous_samples = np.concatenate(anomalous_samples)

    normal_train_size = int(normal_samples.shape[0] * (train_val_test_split[0] / sum(train_val_test_split)))
    normal_val_size = int(normal_samples.shape[0] * (train_val_test_split[1] / sum(train_val_test_split)))
    normal_test_size = int(normal_samples.shape[0] * (train_val_test_split[2] / sum(train_val_test_split)))

    normal_indices = np.arange(normal_samples.shape[0])
    normal_train_indices = np.random.choice(normal_indices, size=normal_train_size, replace=False)
    normal_indices = np.delete(normal_indices, normal_train_indices)
    normal_val_indices = np.random.choice(normal_indices, size=normal_val_size, replace=False)
    normal_test_indices = np.delete(normal_indices, normal_val_indices)

    train_samples = normal_samples[normal_train_indices]

    if anomaly_injection_ratio > 0.0:
        anomaly_indices = np.arange(anomalous_samples.shape[0])
        selected_indices = np.random.choice(anomaly_indices, size=int(train_samples.shape[0] * anomaly_injection_ratio),
                                            replace=False)
        anomaly_indices = np.delete(anomaly_indices, selected_indices)
        train_samples = np.concatenate((train_samples, anomalous_samples[selected_indices]))
        anomalous_samples = anomalous_samples[anomaly_indices]

    np.random.shuffle(train_samples)

    val_samples = np.concatenate(
        (normal_samples[normal_val_indices], anomalous_samples[:int(anomalous_samples.shape[0] * val_anomaly_ratio)]))
    val_labels = np.concatenate((np.zeros(normal_samples[normal_val_indices].shape[0]), np.ones(
        anomalous_samples[:int(anomalous_samples.shape[0] * val_anomaly_ratio)].shape[0])))
    val_indices = np.arange(val_samples.shape[0])
    np.random.shuffle(val_indices)
    val_samples = val_samples[val_indices]
    val_labels = val_labels[val_indices]

    test_samples = np.concatenate(
        (normal_samples[normal_test_indices], anomalous_samples[int(anomalous_samples.shape[0] * val_anomaly_ratio):]))
    test_labels = np.concatenate((np.zeros(normal_samples[normal_test_indices].shape[0]), np.ones(
        anomalous_samples[int(anomalous_samples.shape[0] * val_anomaly_ratio):].shape[0])))
    test_indices = np.arange(test_samples.shape[0])
    np.random.shuffle(test_indices)
    test_samples = test_samples[test_indices]
    test_labels = test_labels[test_indices]

    if verbose:
        print('Train size: ', train_samples.shape)
        print('Validation size: ', val_samples.shape)
        print('Test size: ', test_samples.shape)

    return train_samples, val_samples, val_labels, test_samples, test_labels
