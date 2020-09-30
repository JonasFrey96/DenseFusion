import numpy as np


def vo_cap(D, prec):
    indices = np.where(D != np.inf)
    if len(indices[0]) == 0:
        return 0.0
    D = D[indices]
    prec = prec[indices]
    mrec = np.array([0.0] + D.tolist() + [0.1])
    mprec = np.array([0.0] + prec.tolist() + [prec[-1]])
    for i in range(1, prec.shape[0]):
        mprec[i] = max(mprec[i], mprec[i - 1])
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return np.sum((mrec[i] - mrec[i - 1]) * mprec[i]) * 10


def compute_auc(add_values):
    D = np.array(add_values)

    max_distance = 0.1
    D[np.where(D > max_distance)] = np.inf
    D = np.sort(D)
    N = D.shape[0]
    cumulative = np.cumsum(np.ones((1, N))) / N  # np.arange(0,1+1/N,1/N)
    return vo_cap(D, cumulative) * 100.0


if __name__ == "__main__":
    mu = 0.005
    sigma = 0.00001
    rand = np.abs(np.random.normal(mu, sigma, 100))
    print(rand.shape)
    auc = compute_auc(rand)
    print(auc)
