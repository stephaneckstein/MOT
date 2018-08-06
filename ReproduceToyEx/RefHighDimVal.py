import numpy as np
import time


# This is a general implementation for two period, d-asset MMOT
d = 3
morp = 1


np.random.seed(0)
cov_maty = np.diag(np.random.random_sample(d)) * 2
downscale = 0.5 # affects how much the variance is scaled down from t=1 to t=0
scalemat = np.random.random_sample([d, d])*(1-downscale) + downscale
cov_matx = cov_maty * scalemat
mean = np.random.random_sample(d) * 2 - 1

print(mean)
np.random.seed(int(round(time.time())))

K = sum(mean)
def cost_f(y):
    out = np.maximum(np.sum(y, axis=1) - K, 0)
    return np.sum(morp * out)


def gen_marginal(batch_size):
    # returns sampled marginals of S0, S1 as batch_size x d numpy arrays.
    while True:
        y = np.random.multivariate_normal(size=batch_size, cov=cov_maty, mean=mean)
        yield y

def gen_marginal_comonotone(batch_size):
    while True:
        y = np.zeros([batch_size, d])
        refnormal = np.random.randn(batch_size)
        for i in range(d):
            y[:, i] = refnormal * np.sqrt(cov_maty[i, i]) + mean[i]
        yield y

rval = 0
sample = 1000
batch = 1000
gen = gen_marginal(batch)
for i in range(sample):
    m = next(gen)
    rval += cost_f(m)/(sample * batch)
print(rval)

rvalco = 0
gencomon = gen_marginal_comonotone(batch)
for i in range(sample):
    m = next(gencomon)
    rvalco += cost_f(m)/(sample * batch)
print(rvalco)