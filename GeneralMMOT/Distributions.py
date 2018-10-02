import numpy as np
import time
import matplotlib.pyplot as plt


def gen_OT(batch_size, time_steps, dimension, type='MultiNormal'):
    if type == 'MultiNormal':
        np.random.seed(0)
        sig = np.zeros([time_steps, dimension])
        sig[time_steps-1, :] = np.random.random_sample(dimension) * 2
        print('Sigma Matrix:')
        print(sig)
        sigT = sig[time_steps-1, :]
        while True:
            points = np.random.randn(batch_size, dimension) * sigT
            yield points


# GEN_POINTS
def gen_margs(batch_size, time_steps, dimension, type='Unif0'):
    if type == 'MultiNormal':
        np.random.seed(0)
        sig = np.zeros([time_steps, dimension])
        sig[time_steps-1, :] = np.random.random_sample(dimension) * 2
        downscale = 0.9  # affects how much the variance is scaled down from t to t-1
        for i in range(time_steps-1):
            sig[time_steps-i-2, :] = ((1-downscale / (time_steps-1)) + (downscale / (time_steps-1)) * np.random.random_sample(dimension)) * sig[time_steps-i-1, :]
        np.random.seed(int(round(time.time())))
        print('Sigma Matrix:')
        print(sig)
        while True:
            points = np.random.randn(batch_size, time_steps, dimension) * sig
            yield points

    if type =='2dExtreme':
        sig = np.zeros([time_steps, dimension])
        sig[0, 0] = 0
        sig[0, 1] = 1
        sig[1, 0] = 2
        sig[1, 1] = 1
        print('Sigma Matrix:')
        print(sig)
        while True:
            points = np.random.randn(batch_size, time_steps, dimension) * sig
            yield points

    if type == 'Unif0':
        while True:
            points = np.random.random_sample([batch_size, time_steps, dimension]) * 2 - 1
            points[:, 1, 0] *= 3
            points[:, 1, 1] *= 2
            yield points

    if type == 'Unif1':
        while True:
            points = np.random.random_sample([batch_size, time_steps, dimension]) * 2 - 1
            points[:, 0, 1] *= 2
            points[:, 1, 0] *= 3
            points[:, 1, 1] *= 3
            yield points


def gen_theta(batch_size, time_steps, dimension, type='Unif0'):
    if type == 'MultiNormal':
        np.random.seed(0)
        sig = np.zeros([time_steps, dimension])
        sig[time_steps-1, :] = np.random.random_sample(dimension) * 2
        downscale = 0.9  # affects how much the variance is scaled down from t to t-1
        for i in range(time_steps-1):
            sig[time_steps-i-2, :] = ((1-downscale / (time_steps-1)) + (downscale / (time_steps-1)) * np.random.random_sample(dimension)) * sig[time_steps-i-1, :]
        np.random.seed(int(round(time.time())))
        while True:
            points = np.random.randn(batch_size, time_steps, dimension) * sig
            yield points

    if type =='2dExtreme':
        sig = np.zeros([time_steps, dimension])
        sig[0, 0] = 0
        sig[0, 1] = 1
        sig[1, 0] = 2
        sig[1, 1] = 1
        print('Sigma Matrix:')
        print(sig)
        while True:
            points = np.random.randn(batch_size, time_steps, dimension) * sig
            yield points

    if type == 'Unif0':
        while True:
            points = np.random.random_sample([batch_size, time_steps, dimension]) * 2 - 1
            points[:, 1, 0] *= 3
            points[:, 1, 1] *= 2
            yield points

    if type == 'Unif1':
        while True:
            points = np.random.random_sample([batch_size, time_steps, dimension]) * 2 - 1
            points[:, 0, 1] *= 2
            points[:, 1, 0] *= 3
            points[:, 1, 1] *= 3
            yield points

def gen_comparison(batch_size, time_steps, dimension, type='MultiNormalComon'):
    if type == 'MultiNormalComon':
        np.random.seed(0)
        sig = np.zeros([time_steps, dimension])
        sig[time_steps-1, :] = np.random.random_sample(dimension) * 2
        downscale = 0.9  # affects how much the variance is scaled down from t to t-1
        for i in range(time_steps-1):
            sig[time_steps-i-2, :] = ((1-downscale / (time_steps-1)) + (downscale / (time_steps-1)) * np.random.random_sample(dimension)) * sig[time_steps-i-1, :]
        np.random.seed(int(round(time.time())))
        print('Sigma Matrix:')
        print(sig)
        sigT = sig[time_steps-1, :]
        while True:
            points = np.random.randn(batch_size, dimension)
            points[:, 0] = points[:, 0] * sigT[0]
            for i in range(1, dimension):
                points[:, i] = points[:, i-1] * sigT[i] / sigT[i-1]
            yield points
    if type =='2dExtreme':
        sig = np.zeros([time_steps, dimension])
        sig[0, 0] = 0
        sig[0, 1] = 1
        sig[1, 0] = 2
        sig[1, 1] = 1
        print('Sigma Matrix:')
        print(sig)
        sigT = sig[time_steps-1, :]
        while True:
            points = np.random.randn(batch_size, dimension)
            points[:, 0] = points[:, 0] * sigT[0]
            for i in range(1, dimension):
                points[:, i] = points[:, i-1] * sigT[i] / sigT[i-1]
            yield points



def univ_rebuild(x, w):
    # should take an input of shape (a, b) and return an output of shape (a, c)
    z = np.matmul(x, w[0])
    z = z + w[1]
    a = np.maximum(z, 0)
    z2 = np.matmul(a, w[2]) + w[3]
    a2 = np.maximum(z2, 0)
    z3 = np.matmul(a2, w[4]) + w[5]
    a3 = np.maximum(z3, 0)
    z = np.matmul(a3, w[6]) + w[7]
    return z


def f_rebuild(x, dimension, time_steps, MINMAX=1, ftype='basket'):
    if ftype=='basket':
        return MINMAX * np.maximum(np.sum(x[:, time_steps - 1, :], axis=1) - 0.5 * dimension, 0)
    if ftype=='basket0':
        return MINMAX * np.maximum(np.sum(x[:, time_steps - 1, :], axis=1), 0)


def density_eval(batch_size, x, time_steps, dimension, weights, ftype='basket', batch_up=2 ** 15, MINMAX=1):
    s1_theta = 0
    for i in range(time_steps):
        for j in range(dimension):
            s1_theta += np.sum(univ_rebuild(x[:, i:i + 1, j], w=weights[0][i][j]), axis=1)

    s2_theta = 0
    for i in range(time_steps-1):
        trading_strategy = univ_rebuild(np.reshape(x[:, 0:i + 1, :], [batch_up, (i + 1) * dimension]), weights[1][i])
        increment = np.sum(x[:, i + 1:i + 2, :], axis=1) - np.sum(x[:, i:i + 1, :],axis=1)
        s2_theta += np.sum(trading_strategy * increment, axis=1)

    fvar = f_rebuild(x, dimension, time_steps, ftype=ftype, MINMAX=MINMAX)
    density = np.maximum(fvar - s1_theta - s2_theta, 0)
    return density


def up_sample(batch_size, time_steps, dimension, weights, batch_up=2 ** 16, type='Unif0', ftype='basket', MINMAX=1, multiplier=1, quant=1):
    up_gen = gen_theta(batch_up, time_steps, dimension, type=type)
    sample_part = np.zeros([0, time_steps, dimension])
    while True:
        sample_up = next(up_gen)
        den_val = density_eval(batch_size, sample_up, time_steps, dimension, weights, ftype=ftype, batch_up=batch_up, MINMAX=MINMAX)
        den_max = np.quantile(den_val, q=quant)
        u = np.random.random_sample(batch_up)
        sample_part = np.concatenate([sample_part, sample_up[u * den_max * multiplier <= den_val, :, :]])
        while len(sample_part) >= batch_size:
            sample = sample_part[:batch_size, :, :]
            sample_part = sample_part[batch_size:, :, :]
            yield(sample)
