from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import argparse
from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='ML HW3 - gaussian process')
parser.add_argument('data', help='should be 2_data.mat')

plot_pos = {0:(0, 0), 1:(0, 1), 2:(1, 0), 3:(1, 1)}

def exp_quad_kernel(x, y, params):
    return params[0]*np.exp(-0.5*
                            params[1]*
                            np.subtract.outer(x, y)**2) + \
           params[2] + params[3]*np.multiply.outer(x, y)

def RMSE(x, y):
    return np.sqrt(np.sum((x-y)**2)/len(x))

if __name__ == '__main__':
    args = parser.parse_args()
    data = loadmat(args.data)
    X = data['x'].squeeze()
    T = data['t'].squeeze()
    beta_inv = 1

    train_x, test_x, train_t, test_t = train_test_split(X, T, train_size=60, test_size=40, shuffle=False)

    x = np.linspace(0, 2, 300)
    y = np.empty(300)
    y1 = np.empty(300)
    y2 = np.empty(300)

    parameters = [[1, 4, 0, 0],
                  [0, 0, 0, 1],
                  [1, 4, 0, 5],
                  [1, 64, 10, 0]]

    f, axarr = plt.subplots(2, 2)
    table = PrettyTable(["parameters", "train error", "test error"])

    for p in range(4):
        pos = plot_pos[p]
        C_inv = np.linalg.inv(exp_quad_kernel(train_x, train_x, parameters[p]) + beta_inv * np.identity(60))

        #plot the distribution
        for i in range(300):
            k = exp_quad_kernel(train_x, x[i], parameters[p])
            c = exp_quad_kernel(x[i], x[i], parameters[p]) + beta_inv
            y[i] = np.linalg.multi_dot([k, C_inv, train_t])
            std = np.sqrt(c - np.linalg.multi_dot([k.T, C_inv, k]))
            y1[i] = y[i] + std
            y2[i] = y[i] - std

        axarr[pos].plot(x, y, 'r-')
        axarr[pos].fill_between(x, y1, y2, facecolor='pink', edgecolor='none')
        axarr[pos].scatter(train_x, train_t, facecolors='none', edgecolors='b')
        axarr[pos].set_title(str(parameters[p]))
        axarr[pos].set_xlim(0, 2)
        axarr[pos].set_ylim(-10, 15)
        axarr[pos].set_xlabel('x')
        axarr[pos].set_ylabel('t', rotation=0)

        #calculate the rms on test data
        train_y = np.empty(60)
        for i in range(60):
            k = exp_quad_kernel(train_x, train_x[i], parameters[p])
            c = exp_quad_kernel(train_x[i], train_x[i], parameters[p])
            train_y[i] = np.linalg.multi_dot([k, C_inv, train_t])

        predict = np.empty(40)
        for i in range(40):
            k = exp_quad_kernel(train_x, test_x[i], parameters[p])
            c = exp_quad_kernel(test_x[i], test_x[i], parameters[p])
            predict[i] = np.linalg.multi_dot([k, C_inv, train_t])

        table.add_row(["{"+str(parameters[p])[1:-1]+"}", RMSE(train_y, train_t), RMSE(predict, test_t)])

    print table
    plt.show()
