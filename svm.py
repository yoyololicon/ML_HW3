import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_iris
from svm_sklearn import make_meshgrid
from collections import Counter

def get_w_b(a, t, x):
    at = a*t
    w = at.dot(x)
    Ns = np.count_nonzero(a)
    indice = np.nonzero(a)[0]

    b = np.sum(t[indice]) - np.sum([np.linalg.multi_dot([at[indice], x[indice], x[n]]) for n in indice])
    b /= Ns
    return w, b

def predict(W, B, X, label):
    p = np.empty(len(X))
    for i in range(len(X)):
        candidate = []
        for w, b, lb in zip(W, B, label):
            y = w.dot(X[i]) + b
            if y > 0:
                candidate.append(lb[0])
            else:
                candidate.append(lb[1])
        p[i] = Counter(candidate).most_common(1)[0][0]

    return p

if __name__ == '__main__':
    iris = load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target
    index_0 = np.where(y == 0)
    index_1 = np.where(y == 1)
    index_2 = np.where(y == 2)

    label = ((0, 1), (0, 2), (1, 2))
    t01 = np.concatenate((np.full([50], 1), np.full([50], -1), np.zeros(50)))
    t02 = np.concatenate((np.full([50], 1), np.zeros(50), np.full([50], -1)))
    t12 = np.concatenate((np.zeros(50), np.full([50], 1), np.full([50], -1)))

    #use sklearn to get coefficients
    clf = SVC(kernel='linear', decision_function_shape='ovo')
    clf.fit(X, y)
    coef = np.abs(clf.dual_coef_)   #get alpha
    svi = clf.support_              #get support vector index

    alphas = np.zeros([len(X), 2])
    alphas[svi] = coef.T

    #construct parameter for each classifier
    a01 = np.concatenate((alphas[:100, 0], np.zeros(50)))
    a02 = np.concatenate((alphas[:50, 1], np.zeros(50), alphas[100:, 0]))
    a12 = np.concatenate((np.zeros(50), alphas[50:, 1]))

    w01, b01 = get_w_b(a01, t01, X)
    w02, b02 = get_w_b(a02, t02, X)
    w12, b12 = get_w_b(a12, t12, X)

    #plot the image
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    Z = predict((w01, w02, w12), (b01, b02, b12), np.c_[xx.ravel(), yy.ravel()], label)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)

    plt.scatter(X0[svi], X1[svi], c='black', s=60, label='support vector')
    plt.scatter(X0[index_0], X1[index_0], c='r', s=50, marker='x', label='0')
    plt.scatter(X0[index_1], X1[index_1], c='g', s=50, marker='+', label='1')
    plt.scatter(X0[index_2], X1[index_2], c='b', s=50, marker='*', label='2')
    plt.legend()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xticks(())
    plt.yticks(())

    plt.show()

    #poly kernel
    clf2 = SVC(kernel='poly', degree=2, decision_function_shape='ovo')
    clf2.fit(X, y)

    poly2_X = np.vstack((X[:, 0]**2, np.sqrt(2)*X[:, 0]*X[:, 1], X[:, 1]**2)).T
    coef = np.abs(clf2.dual_coef_)   #get alpha
    svi = clf2.support_              #get support vector index

    alphas = np.zeros([len(X), 2])
    alphas[svi] = coef.T

    #construct parameter for each classifier
    a01 = np.concatenate((alphas[:100, 0], np.zeros(50)))
    a02 = np.concatenate((alphas[:50, 1], np.zeros(50), alphas[100:, 0]))
    a12 = np.concatenate((np.zeros(50), alphas[50:, 1]))

    w01, b01 = get_w_b(a01, t01, poly2_X)
    w02, b02 = get_w_b(a02, t02, poly2_X)
    w12, b12 = get_w_b(a12, t12, poly2_X)

    Z = predict((w01, w02, w12), (b01, b02, b12),
                np.vstack((xx.ravel()**2, np.sqrt(2)*xx.ravel()*yy.ravel(), yy.ravel()**2)).T, label)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)

    plt.scatter(X0[svi], X1[svi], c='black', s=60, label='support vector')
    plt.scatter(X0[index_0], X1[index_0], c='r', s=50, marker='x', label='0')
    plt.scatter(X0[index_1], X1[index_1], c='g', s=50, marker='+', label='1')
    plt.scatter(X0[index_2], X1[index_2], c='b', s=50, marker='*', label='2')
    plt.legend()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xticks(())
    plt.yticks(())

    plt.show()