import os
import sys

from tqdm.std import tqdm
from torch.utils.data import DataLoader

sys.path.append('../')

from src.data import prepare_data
from src.data import ECGDataset


def test_prepare_data():
    print(os.listdir('./'))
    prepare_data('data/mit_ecg_processed/', anomaly_injection_ratio=0.1)


def test_ecgdataset():
    train_samples, val_samples, val_labels, test_samples, test_labels = prepare_data('data/mit_ecg_processed/',
                                                                                     anomaly_injection_ratio=0.1)
    train_data = ECGDataset(train_samples)
    test_data = ECGDataset(test_samples, test_labels)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=1, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=1, drop_last=True)

    # for x in tqdm(train_loader):
    #     print(x.shape)
    #
    # for x, y in tqdm(test_loader):
    #     print(x.shape, y.shape)


# def test_apply_transformation():
#     train_samples, val_samples, val_labels, test_samples, test_labels = prepare_data('./data/mit_ecg_processed/',
#                                                                                      anomaly_injection_ratio=0.1)
#
#     trans_samples, trans_labels = apply_transformation(train_samples, 32)
#     assert False, trans_labels
