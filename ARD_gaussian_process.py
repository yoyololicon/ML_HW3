from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import argparse
from prettytable import PrettyTable
from gaussian_process import exp_quad_kernel, RMSE
from bayesian_linear_regression import blr

parser = argparse.ArgumentParser(description='ML HW3 - ARD framework of GP')
parser.add_argument('data', help='ex: Dataset/Problem1/2_data.mat')

def dev_log_like(C_inv, C_dev, t):
    return -0.5 * np.trace(C_inv.dot(C_dev)) + \
           0.5 * np.linalg.multi_dot([t.T, C_inv, C_dev, C_inv, t])

if __name__ == '__main__':
    args = parser.parse_args()
    data = loadmat(args.data)
    X = data['x'].squeeze()
    T = data['t'].squeeze()
    beta_inv = 1

    train_x, test_x, train_t, test_t = train_test_split(X, T, train_size=60, test_size=40, shuffle=False)

    parameters = [[3, 6, 4, 5]]
    dev_func = [0, 0, 0, 0]
    learning_rate = 0.001

    table = PrettyTable(["optimal parameters", "train error", "test error"])

    while True:
        C_inv = np.linalg.inv(exp_quad_kernel(train_x, train_x, parameters[-1]) + beta_inv * np.identity(60))

        #update parameter
        dev_func[0] = \
            dev_log_like(C_inv,
                         np.exp(-0.5 *
                                parameters[-1][1] *
                                np.subtract.outer(train_x, train_x)**2),
                         train_t)
        dev_func[1] = \
            dev_log_like(C_inv,
                         parameters[-1][0] *
                         -0.5 *
                         np.subtract.outer(train_x, train_x) *
                         np.exp(-0.5 *
                                parameters[-1][1] *
                                np.subtract.outer(train_x, train_x)**2)
                         ,
                         train_t)
        dev_func[2] = dev_log_like(C_inv, np.full([60, 60], 1), train_t)
        dev_func[3] = dev_log_like(C_inv, np.multiply.outer(train_x, train_x), train_t)
        parameters.append([p + learning_rate * dev for p, dev in zip(parameters[-1], dev_func)])

        if np.max(np.abs(dev_func)) < 6:
            break

    params = np.array(parameters)
    plt.plot(params[:, 0], label='hyperparameter 0')
    plt.plot(params[:, 1], label='hyperparameter 1')
    plt.plot(params[:, 2], label='hyperparameter 2')
    plt.plot(params[:, 3], label='hyperparameter 3')
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("value", rotation=0)
    plt.show()

    #plot distribution
    x = np.linspace(0, 2, 300)
    y = np.empty(300)
    y1 = np.empty(300)
    y2 = np.empty(300)

    C_inv = np.linalg.inv(exp_quad_kernel(train_x, train_x, parameters[-1]) + beta_inv * np.identity(60))

    for i in range(300):
        k = exp_quad_kernel(train_x, x[i], parameters[-1])
        c = exp_quad_kernel(x[i], x[i], parameters[-1]) + beta_inv
        y[i] = np.linalg.multi_dot([k, C_inv, train_t])
        std = np.sqrt(c - np.linalg.multi_dot([k.T, C_inv, k]))
        y1[i] = y[i] + std
        y2[i] = y[i] - std

    # calculate the rms on training and test data
    train_y = np.empty(60)
    for i in range(60):
        k = exp_quad_kernel(train_x, train_x[i], parameters[-1])
        train_y[i] = np.linalg.multi_dot([k, C_inv, train_t])

    predict = np.empty(40)
    for i in range(40):
        k = exp_quad_kernel(train_x, test_x[i], parameters[-1])
        predict[i] = np.linalg.multi_dot([k, C_inv, train_t])

    table.add_row(["{" + str([round(p, 6) for p in parameters[-1]])[1:-1] + "}", RMSE(train_y, train_t), RMSE(predict, test_t)])
    print "For ARD gaussian process"
    print table

    fig, ax = plt.subplots(1, 2, sharex='row')

    ax[0].plot(x, y, 'r-')
    ax[0].fill_between(x, y1, y2, facecolor='pink', edgecolor='none')
    ax[0].scatter(train_x, train_t, facecolors='none', edgecolors='b')
    ax[0].set_title(str([round(p, 6) for p in parameters[-1]]))
    ax[0].set_xlim(0, 2)
    ax[0].set_ylim(-10, 15)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t', rotation=0)

    train_y_blr, std_blr = blr(train_x, train_t, train_x)
    predict_blr, std_blr = blr(train_x, train_t, test_x)

    table2 = PrettyTable(["train error", "test error"])
    table2.add_row([RMSE(train_y_blr, train_t), RMSE(predict_blr, test_t)])
    print "For bayesian linear regression"
    print table2

    y, std_blr = blr(train_x, train_t, x)
    y1 = y + std_blr
    y2 = y - std_blr

    ax[1].plot(x, y, 'r-')
    ax[1].fill_between(x, y1, y2, facecolor='pink', edgecolor='none')
    ax[1].scatter(train_x, train_t, facecolors='none', edgecolors='b')
    ax[1].set_title("M = 7, s = 0.1, alpha = 0.000001, beta = 1")
    ax[1].set_xlim(0, 2)
    ax[1].set_ylim(-10, 15)
    ax[1].set_xlabel('x')
    plt.show()