from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
from prettytable import PrettyTable
from gaussian_process import exp_quad_kernel, RMSE

parser = argparse.ArgumentParser(description='ML HW3')
parser.add_argument('data', help='should be a matlab file.')

def dev_log_like(C_inv, C_inv_dev, t):
    return -0.5 * np.trace(C_inv.dot(C_inv_dev)) + \
           0.5 * np.linalg.multi_dot([t.T, C_inv, C_inv_dev, C_inv, t])

if __name__ == '__main__':
    args = parser.parse_args()
    data = loadmat(args.data)
    X = data['x'].squeeze()
    T = data['t'].squeeze()

    train_x, test_x, train_t, test_t = train_test_split(X, T, train_size=60, test_size=40, shuffle=False)

    parameters = [3, 6, 4, 5]
    dev_func = [0, 0, 0, 0]
    learning_rate = 0.00001

    
    while True:
        C_inv = np.linalg.inv(exp_quad_kernel(train_x, train_x, parameters) + np.identity(60))

        #update parameter
        dev_func[0] = \
            dev_log_like(C_inv,
                         np.exp(-0.5 *
                                parameters[1] *
                                np.subtract.outer(train_x, train_x)),
                         train_t)
        dev_func[1] = \
            dev_log_like(C_inv,
                         parameters[0] *
                         np.exp(-0.5 *
                                parameters[1] *
                                np.subtract.outer(train_x, train_x)) *
                         -0.5 *
                         np.subtract.outer(train_x, train_x),
                         train_t)
        dev_func[2] = dev_log_like(C_inv, np.full([60, 60], 1), train_t)
        dev_func[3] = dev_log_like(C_inv, np.multiply.outer(train_x, train_x), train_t)

        if max(dev_func) < 6 :
            break
        print dev_func

        parameters = [parameters[j] + learning_rate * dev_func[j] for j in range(4)]


