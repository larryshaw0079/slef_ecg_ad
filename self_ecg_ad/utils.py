import abc
import pdb

import torch
import numpy as np
from tqdm.std import tqdm

from scipy.special import psi, polygamma
from sympy import Matrix, GramSchmidt


def calc_approx_alpha_sum(observations):
    N = len(observations)
    f = np.mean(observations, axis=0)

    return (N * (len(f) - 1) * (-psi(1))) / (
            N * np.sum(f * np.log(f)) - np.sum(f * np.sum(np.log(observations), axis=0)))


def inv_psi(y, iters=5):
    # initial estimate
    cond = y >= -2.22
    x = cond * (np.exp(y) + 0.5) + (1 - cond) * -1 / (y - psi(1))

    for _ in range(iters):
        x = x - (psi(x) - y) / polygamma(1, x)
    return x


def fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=1000):
    alpha_new = alpha_old = alpha_init
    for _ in range(max_iter):
        alpha_new = inv_psi(psi(np.sum(alpha_old)) + log_p_hat)
        if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-9:
            break
        alpha_old = alpha_new
    return alpha_new


def dirichlet_normality_score(alpha, p):
    return np.sum((alpha - 1) * np.log(p), axis=-1)


def normality_score(observations, predictions):
    log_observations = np.log(observations).mean(axis=0)
    alpha_sum_approx = calc_approx_alpha_sum(observations)
    alpha_0 = observations.mean(axis=0)*alpha_sum_approx

    mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_observations)
    result = dirichlet_normality_score(mle_alpha_t, predictions)

    return result


def simplified_normality_score(predictions):
    scores = predictions.mean(axis=-1)
    return scores


def trans_flip(x, flip=True):
    if flip:
        return np.flip(x, axis=-1)
    else:
        return x


def trans_roll(x, dir, shift):
    return np.roll(x, shift=dir*shift, axis=-1)


def basis_normalize(X):
    assert X.ndim == 2
    vlist = [Matrix(X[i, :]) for i in range(len(X))]
    out = GramSchmidt(vlist, False)

    return np.concatenate([np.array(out[i].tolist()).astype(np.float32).reshape(-1, 1) for i in range(len(vlist))])


# class Transformation(abc.ABC):
#     def __init__(self):
#         pass
#
#     @abc.abstractmethod
#     def apply_transformation(self, x, ind):
#         pass


class GeometricTransformation(object):
    def __init__(self):
        self.trans_list = []

    def apply_transformation(self, x, ind):
        assert x.ndim == 2
        return self.trans_list[ind](x)


class RandomTransformation(object):
    def __init__(self, length, num_trans, normalize=True, gpu=False):
        self.trans_mats = np.random.randn(num_trans, length, length).astype(np.float32)
        self.num_trans = num_trans
        
        if normalize:
            for t in tqdm(range(len(self.trans_mats))):
                self.trans_mats[t] = basis_normalize(self.trans_mats[t])
                
        self.trans_mats = torch.from_numpy(self.trans_mats)
        if gpu:
            self.trans_mats = self.trans_mats.cuda()

    def apply_transformation_single(self, x, ind):
        return x@self.trans_mats[ind]

    def apply_transformation_all(self, x):
        out = torch.cat([self.apply_transformation_single(x, i) for i in range(self.num_trans)])
        return out
