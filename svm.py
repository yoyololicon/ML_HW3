import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from svm_sklearn import make_meshgrid
from collections import Counter
import argparse

parser = argparse.ArgumentParser(description='ML HW3 - svm')

def get_w_b(a, t, x):
    at = a*t
    w = at.dot(x)
    Ns = np.count_nonzero(a)
    indice = np.nonzero(a)[0]

    b = np.sum(t[indice]) - np.sum(np.linalg.multi_dot([at[indice], x[indice], x[indice].T]))
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
    # Take the first two features.
    X = iris.data[:, :2]
    y = iris.target
    index_0 = np.where(y == 0)
    index_1 = np.where(y == 1)
    index_2 = np.where(y == 2)
    #plot 4 images
    fig, sub = plt.subplots(2, 2)

    #make label clearly
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

    #plot region
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    Z = predict((w01, w02, w12), (b01, b02, b12), np.c_[xx.ravel(), yy.ravel()], label)
    Z = Z.reshape(xx.shape)

    sub[0][0].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    sub[0][0].scatter(X0[svi], X1[svi], c='black', s=60, label='support vector')
    sub[0][0].scatter(X0[index_0], X1[index_0], c='r', s=50, marker='x', label='0')
    sub[0][0].scatter(X0[index_1], X1[index_1], c='g', s=50, marker='+', label='1')
    sub[0][0].scatter(X0[index_2], X1[index_2], c='b', s=50, marker='*', label='2')
    sub[0][0].legend()
    sub[0][0].set_xlim(xx.min(), xx.max())
    sub[0][0].set_ylim(yy.min(), yy.max())
    sub[0][0].set_xlabel('Sepal length')
    sub[0][0].set_ylabel('Sepal width')
    sub[0][0].set_title('linear kernel')

    #poly kernel
    clf2 = SVC(kernel='poly', degree=2, decision_function_shape='ovo')
    clf2.fit(X, y)

    #transform to polynomial form
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
    sub[0][1].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    sub[0][1].scatter(X0[svi], X1[svi], c='black', s=60, label='support vector')
    sub[0][1].scatter(X0[index_0], X1[index_0], c='r', s=50, marker='x', label='0')
    sub[0][1].scatter(X0[index_1], X1[index_1], c='g', s=50, marker='+', label='1')
    sub[0][1].scatter(X0[index_2], X1[index_2], c='b', s=50, marker='*', label='2')
    sub[0][1].legend()
    sub[0][1].set_xlim(xx.min(), xx.max())
    sub[0][1].set_ylim(yy.min(), yy.max())
    sub[0][1].set_xlabel('Sepal length')
    sub[0][1].set_ylabel('Sepal width')
    sub[0][1].set_title('polynomial kernel')

    #use LDA to reduce dimension, similar to PCA
    #calculate mean
    X = iris.data
    means = np.mean(X.reshape(-1, 50, 4), axis=1)
    overall_mean = np.mean(X, axis=0)

    #calculate S_w and S_b
    S_w = np.sum([(X[i*50:(i+1)*50] - means[i]).T.dot(X[i*50:(i+1)*50] - means[i]) for i in range(3)], axis=0)
    S_b = 50*(means - overall_mean).T.dot(means - overall_mean)

    #this part I ectually copy from my IML HW 2 PCA part
    mat = np.linalg.inv(S_w).dot(S_b)
    va, ve = np.linalg.eig(mat)
    eig_pairs = [(va[i], ve[:, i]) for i in range(len(va))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    w = np.vstack((eig_pairs[0][1], eig_pairs[1][1]))

    #reduce dimension of the data
    lda_X = X.dot(w.T)

    #re-train
    clf.fit(lda_X, y)
    coef = np.abs(clf.dual_coef_)  # get alpha
    svi = clf.support_  # get support vector index

    alphas = np.zeros([len(lda_X), 2])
    alphas[svi] = coef.T

    # construct parameter for each classifier
    a01 = np.concatenate((alphas[:100, 0], np.zeros(50)))
    a02 = np.concatenate((alphas[:50, 1], np.zeros(50), alphas[100:, 0]))
    a12 = np.concatenate((np.zeros(50), alphas[50:, 1]))

    w01, b01 = get_w_b(a01, t01, lda_X)
    w02, b02 = get_w_b(a02, t02, lda_X)
    w12, b12 = get_w_b(a12, t12, lda_X)

    X0, X1 = lda_X[:, 0], lda_X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    Z = predict((w01, w02, w12), (b01, b02, b12), np.c_[xx.ravel(), yy.ravel()], label)
    Z = Z.reshape(xx.shape)

    sub[1][0].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    sub[1][0].scatter(X0[svi], X1[svi], c='black', s=60, label='support vector')
    sub[1][0].scatter(X0[index_0], X1[index_0], c='r', s=50, marker='x', label='0')
    sub[1][0].scatter(X0[index_1], X1[index_1], c='g', s=50, marker='+', label='1')
    sub[1][0].scatter(X0[index_2], X1[index_2], c='b', s=50, marker='*', label='2')
    sub[1][0].legend()
    sub[1][0].set_xlim(xx.min(), xx.max())
    sub[1][0].set_ylim(yy.min(), yy.max())
    sub[1][0].set_xlabel('Dimension 1 after LDA(2)')
    sub[1][0].set_ylabel('Dimension 2 after LDA(2)')

    #re-train polynomial model
    clf2.fit(lda_X, y)

    poly2_X = np.vstack((lda_X[:, 0]**2, np.sqrt(2)*lda_X[:, 0]*lda_X[:, 1], lda_X[:, 1]**2)).T
    coef = np.abs(clf2.dual_coef_)   #get alpha
    svi = clf2.support_              #get support vector index

    alphas = np.zeros([len(lda_X), 2])
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
    sub[1][1].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    sub[1][1].scatter(X0[svi], X1[svi], c='black', s=60, label='support vector')
    sub[1][1].scatter(X0[index_0], X1[index_0], c='r', s=50, marker='x', label='0')
    sub[1][1].scatter(X0[index_1], X1[index_1], c='g', s=50, marker='+', label='1')
    sub[1][1].scatter(X0[index_2], X1[index_2], c='b', s=50, marker='*', label='2')
    sub[1][1].legend()
    sub[1][1].set_xlim(xx.min(), xx.max())
    sub[1][1].set_ylim(yy.min(), yy.max())
    sub[1][1].set_xlabel('Dimension 1 after LDA(2)')
    sub[1][1].set_ylabel('Dimension 2 after LDA(2)')
    plt.show()