import numpy as np
from sklearn.mixture import GaussianMixture


def Lap_update(good_samples, n_comp=40, cov_type='full'):
    # returns a generator function that generates samples from a Laplace approximation of points in good_samples


    print('Fitting mixture of Gaussians ... ')
    n, dim = good_samples.shape

    if n < n_comp:
        n_comp = n

    estimator = GaussianMixture(n_components=n_comp,
                    covariance_type=cov_type, max_iter=2500, random_state=0)
    estimator.means_init = [np.random.random_sample(dim)
                            for i in range(n_comp)]
    estimator.fit(good_samples)
    print('Done!')

    def gen_lap(batch_size):
        while True:
            yield estimator.sample(batch_size)[0]

    return gen_lap

# good_samples = np.ones([500, 2])
# g_fun = Lap_update(good_samples)
# gen = g_fun(10)
# samp = next(gen)
# print(samp)