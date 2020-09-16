import abc
from typing import List

import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline


class BaseTransform(abc.ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def get_params(self):
        return self.__dict__

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)


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
        # x shape: (batch, channel, len)
        # noise shape: (1, 1, len)
        # Broadcast to each channel and sample
        return x + np.random.normal(loc=0, scale=self.__sigma, size=(1, 1, x.shape[2]))


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
        return x[:, :, ::-1] if self.__is_flip else x


class Permutation(BaseTransform):
    """

    Parameters
    ----------
    x_length :
    permutation : np.random.permutation(np.arange(num_segment))
    """

    def __init__(self, x_length: int, permutation: np.ndarray):
        super(Permutation, self).__init__()
        self.__length = x_length
        self.__permutation = permutation
        assert self.__permutation.ndim == 1
        self.__num_segment = len(self.__permutation)
        assert self.__length % self.__num_segment == 0
        self.__len_segment = self.__length // self.__num_segment
        self.__inds = np.tile(np.arange(self.__len_segment), self.__num_segment)
        self.__inds = self.__inds + np.concatenate(
            [np.full(shape=self.__len_segment, fill_value=i * self.__len_segment) for i in self.__permutation])

    def __call__(self, x: np.ndarray):
        return x[:, :, self.__inds]


class TimeWrap(BaseTransform):
    def __init__(self, base_curve: np.ndarray):
        super(TimeWrap, self).__init__()

        self.__base_curve = base_curve

    @staticmethod
    def random_curve(x_length: int, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
        xx = np.arange(0, x_length, (x_length - 1) / (knot + 1))
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
        x_range = np.arange(x_length)
        cs_x = CubicSpline(xx, yy)

        return np.array(cs_x(x_range))

    def __distort_timestep(self, x_length: int) -> np.ndarray:
        timestamp_cumsum = np.cumsum(self.__base_curve)  # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        timestamp_cumsum = timestamp_cumsum * (x_length - 1) / timestamp_cumsum[-1]

        return timestamp_cumsum

    def __time_wrap(self, x: np.ndarray) -> np.ndarray:
        new_timestamp = self.__distort_timestep(x.shape[-1])
        x_new = np.zeros(x.shape)
        x_range = np.arange(x.shape[-1])

        @njit
        def process(x_new_, x_range_, new_timestamp_, x_):
            for i in range(x_new_.shape[0]):
                for j in range(x_new_.shape[1]):
                    x_new_[i, j] = np.interp(x_range_, new_timestamp_, x_[i, j])
            return x_new_

        return process(x_new, x_range, new_timestamp, x)

    def __call__(self, x: np.ndarray):
        return self.__time_wrap(x)
