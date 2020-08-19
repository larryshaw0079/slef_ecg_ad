import sys
import pytest

sys.path.append('..')

from preprocess_ecg import process_patient, prepare_data


def test_process_patient():
    samples, labels = process_patient(100, './data/mit_ecg', 140, 180)
    assert False, '{}, {}'.format(samples[0].shape, labels[0])


def test_prepare_data():
    assert False
