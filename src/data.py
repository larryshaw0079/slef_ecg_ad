import os

import numpy as np
import scipy.io as sio
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, root: str, label=None, normalize=True, num_channel=2):
        mat_data = sio.loadmat(os.path.join(root, 'mit_ecg.mat'))
        self.data = mat_data['data']
        self.label = mat_data['label']

        if normalize:
            self.data = MinMaxScaler().fit_transform(self.data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        x = torch.from_numpy(self.data[item].astype(np.float32))
        if self.label is not None:
            y = torch.from_numpy(self.label[item].astype(np.int))
            return x, y
        else:
            return x
