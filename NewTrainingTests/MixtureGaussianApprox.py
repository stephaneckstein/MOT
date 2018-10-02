import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold


x = np.loadtxt('Gam1000_1_sample_pihat.txt')
d = 2
print(x.shape)

# iris = datasets.load_iris()
# X_train = iris.data
# print(X_train)


plt.scatter(x[:, 0], x[:, 1])
plt.show()


N_COMP = 10

estimators = dict((cov_type, GaussianMixture(n_components=N_COMP,
                   covariance_type=cov_type, max_iter=500, random_state=0))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])

for index, (name, estimator) in enumerate(estimators.items()):
    estimator.means_init = [np.random.random_sample(d)
                                    for i in range(N_COMP)]

    # Train the other parameters using the EM algorithm.
    estimator.fit(x)
    print(estimator)
    xe = estimator.sample(1000)[0]
    plt.scatter(xe[:, 0], xe[:, 1])
    plt.show()