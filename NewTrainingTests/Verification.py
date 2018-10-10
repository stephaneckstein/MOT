import numpy as np


def verify_marginal(m1, m2, strikes):
    # m1 and m2 are one samples from one dimensional distributions.
    # We compare call prices for both marginals for strike values in the list strikes

    diffs = []
    for s in strikes:
        diff = np.abs(np.mean(np.maximum(m1-s, 0)) - np.mean(np.maximum(m2-s, 0)))
        diffs.append(diff)

    av = np.mean(diffs)
    return av


def verify_martingale(X, Y, listoffuns):
    # X and Y are [n, d] numpy arrays (n samples from a d-dimensional distribution)
    # tests whether |E[<h(X) , Y-X>]| is small, where < , > is the R^d scalar product.
    # tested for all functions h in listoffuns, where h maps R^d to R^d
    # TODO adjust for functions that handle [n, d] arrays ...

    n, d = X.shape

    vals = []
    for h in listoffuns:
        val = 0
        for i in range(n):
            val += np.sum(h(X[i, :]) * (Y[i, :] - X[i, :]))
        val /= n
        vals.append(np.abs(val))
    return np.mean(vals)



m1 = np.random.randn(10000)
m2 = np.random.randn(100)
strikes = np.arange(-2, 2, 0.1)
print(verify_marginal(m1, m2, strikes))

m2 = np.random.randn(10000)
print(verify_marginal(m1, m2, strikes))


X = np.random.randn(10000, 2)
Y = np.random.randn(10000, 2) * 1.5
funs = []
for i in range(20):
    for j in range(20):
        def h(x):
            S = np.ones(2)
            S[0] = (i-10)/10
            S[1] = (j-10)/10
            return np.maximum(x-S, 0)
        funs.append(h)

print(funs)
print(verify_martingale(X, Y, funs))