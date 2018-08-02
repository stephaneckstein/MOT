import numpy as np
import quadprog as qp
import cvxopt
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import matrix
import picos as pic
import mosek
import time


startt = time.time()
# Example 1 from Alfonsi et al paper
I = 50  # number of samples
J = 50
mu = np.random.randn(I)
nu = np.random.randn(J) * np.sqrt(1.1)

# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# sns.distplot(mu, ax=ax1, bins=np.arange(-3, 3.25, 0.25))
# sns.distplot(nu, ax=ax2, bins=np.arange(-3, 3.25, 0.25))
# plt.show()


def convexify(mu, nu):
    # mu, nu are I x d respective J x d size np arrays
    # I, J are size of samples, d is dimension of dists
    if len(mu.shape)>1:
        (I, d) = mu.shape
    else:
        I = len(mu)
        d = 1
    J = len(nu)

    prob = pic.Problem()
    q = prob.add_variable('x', size=[I, J])

    obj = 0
    if d == 17:
        # For some reason that doesn't work...
        obj = 1/I * pic.sum([(mu[i] - pic.sum([q[i, j] * nu[j] for j in range(J)], 'j', '[J]'))**2 for i in range(I)], 'i', '[I]')
    elif d==1:
        for i in range(I):
            obj += 1/I * mu[i] ** 2

        for i in range(I):
            for j in range(J):
                obj -= 2/I * q[i, j] * mu[i] * nu[j]

        for i in range(I):
            print(i)
            for j in range(J):
                for j2 in range(J):
                    obj += 1/I * q[i, j] * q[i, j2] * nu[j] * nu[j2]
    else:
        print('Haste noch net implementiert Junge')
        return 0
    for i in range(I):
        for j in range(J):
            prob.add_constraint(q[i, j] > 0)

    for i in range(I):
        prob.add_constraint(pic.sum([q[i, j] for j in range(J)], 'j', '0...(J-1)') == 1)

    for j in range(J):
        prob.add_constraint(pic.sum([q[i, j] for i in range(J)], 'i', '0...(I-1)') == I/J)

    prob.set_objective('min', obj)
    print(prob)
    sol = prob.solve(solver='mosek')


    qv = q.value
    new_mu = np.zeros(shape=mu.shape)
    for i in range(I):
        for j in range(J):
            if d > 1:
                new_mu[i, :] += qv[i, j] * nu[j, :]
            else:
                new_mu[i] += qv[i, j] * nu[j]

    return new_mu, qv

new_mu, q = convexify(mu, nu)
print(startt - time.time())
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
sns.distplot(mu, ax=ax1, bins=np.arange(-3, 3.25, 0.25))
sns.distplot(nu, ax=ax2, bins=np.arange(-3, 3.25, 0.25))
sns.distplot(new_mu, ax=ax3, bins=np.arange(-3, 3.25, 0.25))
print(np.mean(mu))
print(np.mean(nu))
print(np.mean(new_mu))
plt.show()

