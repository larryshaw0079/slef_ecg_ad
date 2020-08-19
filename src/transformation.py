import itertools
from typing import Union

import numpy as np
from tqdm.std import tqdm

import torch


class Transformation(object):
    def __init__(self, image_width, image_height):
        self.flip_params = [True, False]
        self.translation_x_params = [0, -int(image_width*0.25), int(image_width*0.25)]
        self.translation_y_params = [0, -int(image_height*0.25), int(image_height*0.25)]
        self.rotation_params = [0, 1, 2, 3]

        self.param_list = itertools.product(*tuple([self.flip_params, self.translation_x_params,
                                                   self.translation_y_params, self.rotation_params]))
        self.num_transformation = len(self.flip_params)*len(self.translation_x_params)*len(self.translation_y_params)*len(self.rotation_params)

    @staticmethod
    def transform_gaussian_noise(x: np.ndarray):
        return x + np.random.randn(x.shape)

    @staticmethod
    def transform_scaling(x: np.ndarray, factor: float=1.0):
        return x*factor

    @staticmethod
    def transform_negation(x: np.ndarray):
        return x*-1

    @staticmethod
    def transform_flipping(x: np.ndarray):
        return x[:,::-1]

    @staticmethod
    def transform_permutation(x: np.ndarray):
        pass

    @staticmethod
    def transform_time_wrapping(x: np.ndarray):
        pass

    def transform_batch(self, x: Union[np.ndarray, torch.Tensor]):
        if isinstance(x, np.ndarray):
            results = np.zeros((x.shape[0], self.num, *x.shape[1:]))

            for i in tqdm(range(x.shape[0]), desc='TRANSFORMATION'):
                for j, (is_flip, tx, ty, k_rotate) in enumerate(self.param_list):
                    results[i, j] = self.transform_array(x[i], is_flip, k_rotate, tx, ty)

            return results.reshape(-1, *results.shape[-3:]), np.tile(np.arange(self.num), x.shape[0])
        elif isinstance(x, torch.Tensor):
            results = torch.zeros((x.shape[0], self.num, *x.shape[1:]))

            for i in tqdm(range(x.shape[0]), desc='TRANSFORMATION'):
                for j, (is_flip, tx, ty, k_rotate) in enumerate(self.param_list):
                    results[i, j] = self.transform_tensor(x[i], is_flip, k_rotate, tx, ty)

            return results.reshape(-1, *results.shape[-3:]), torch.arange(self.num).repeat(x.shape[0])
        else:
            raise ValueError
