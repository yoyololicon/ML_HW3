import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def get_posterior(a, b, XX, t):
    D = len(XX[0, :])
    SN = np.linalg.inv(a*np.identity(D) + b*np.dot(XX.T, XX))
    mN = b*np.linalg.multi_dot([SN, XX.T, t])
    return mN, SN

def get_basis_form(u, X, s):
    m = len(X)
    n = len(u)
    transform = np.empty([m, n])
    for i in range(m):
        transform[i] = np.array([sigmoid((X[i]-u[j])/s) for j in range(n)])
    return transform

def blr(train_x, train_t, test_x):
    M = 7
    s = 0.1
    u = [float(j*2)/M for j in range(M)]
    alpha = 1./math.pow(10, 6)
    beta = 1

    post_mean, post_var = get_posterior(alpha, beta, get_basis_form(u, train_x, s), train_t)
    test_y = np.empty(len(test_x))
    test_std = np.empty(len(test_x))
    for i in range(len(test_y)):
        trans_x = np.squeeze(get_basis_form(u, np.array([test_x[i]]), s))
        test_y[i] = np.dot(post_mean.T, trans_x)
        test_std[i] = math.sqrt(1./beta + np.linalg.multi_dot([trans_x.T, post_var, trans_x]))

    return test_y, test_std
