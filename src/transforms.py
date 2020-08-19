import itertools
import abc
from typing import Union, List

import numpy as np
from scipy.interpolate import CubicSpline
from tqdm.std import tqdm

import torch



class BaseTransform(abc.ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class Compose(BaseTransform):
    def __init__(self, trans_list: List[BaseTransform]):
        super(Compose, self).__init__()
        self.__trans_list = trans_list

    def __call__(self, x: np.ndarray):
        for trans in self.__trans_list:
            x = trans(x)
        return x


class GaussianNoise(BaseTransform):
    def __init__(self, sigma: float=1.0):
        super(GaussianNoise, self).__init__()
        self.__sigma = sigma

    def __call__(self, x: np.ndarray):
        return x + np.random.normal(loc=0, scale=self.__sigma, size=x.shape)


class Scale(BaseTransform):
    def __init__(self, factor: float=1.0):
        super(Scale, self).__init__()
        self.__factor = factor

    def __call__(self, x: np.ndarray):
        return x*self.__factor


class Negation(BaseTransform):
    def __init__(self):
        super(Negation, self).__init__()

    def __call__(self, x: np.ndarray):
        return x*-1


class Flip(BaseTransform):
    def __init__(self, is_flip: bool=True):
        super(Flip, self).__init__()
        self.__is_flip = is_flip

    def __call__(self, x: np.ndarray):
        return x[:,:,::-1] if self.__is_flip else x


class Permutation(BaseTransform):
    """

    Parameters
    ----------
    length :
    permutation : np.random.permutation(np.arange(num_segment))
    """
    def __init__(self, length: int, permutation: List[int]):
        super(Permutation, self).__init__()
        self.__length = length
        self.__permutation = permutation
        self.__num_segment = len(self.__permutation)
        assert self.__length % self.__num_segment == 0
        self.__len_segment = self.__length // self.__num_segment
        self.__inds = np.tile(np.arange(self.__len_segment), self.__num_segment)
        self.__inds = self.__inds + np.concatenate([np.full(shape=self.__len_segment, fill_value=i*self.__len_segment) for i in self.__permutation])

    def __call__(self, x: np.ndarray):
        return x[:, :, self.__inds]


class TimeWrap(BaseTransform):
    def __init__(self):
        super(TimeWrap, self).__init__()

    @staticmethod
    def random_curve(X, sigma=0.2, knot=4):
        xx = (np.ones((X.shape[1], 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
        x_range = np.arange(X.shape[0])
        cs_x = CubicSpline(xx[:, 0], yy[:, 0])
        cs_y = CubicSpline(xx[:, 1], yy[:, 1])
        cs_z = CubicSpline(xx[:, 2], yy[:, 2])
        return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()

    @staticmethod
    def distort_timestep(X, sigma=0.2):
        tt = TimeWrap.random_curve(X, sigma)  # Regard these samples aroun 1 as time intervals
        tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        t_scale = [(X.shape[0] - 1) / tt_cum[-1, 0], (X.shape[0] - 1) / tt_cum[-1, 1], (X.shape[0] - 1) / tt_cum[-1, 2]]
        tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
        tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
        tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
        return tt_cum

    @staticmethod
    def time_wrap(X, sigma=0.2):
        tt_new = TimeWrap.distort_timestep(X, sigma)
        X_new = np.zeros(X.shape)
        x_range = np.arange(X.shape[0])
        X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
        X_new[:, 1] = np.interp(x_range, tt_new[:, 1], X[:, 1])
        X_new[:, 2] = np.interp(x_range, tt_new[:, 2], X[:, 2])
        return X_new

    def __call__(self, x: np.ndarray):
        pass


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
