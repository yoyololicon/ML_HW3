import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

if __name__ == '__main__':
    iris = load_iris()
    # Take the first two features.
    X = iris.data[:, :2]
    y = iris.target
    index_0 = np.where(y == 0)
    index_1 = np.where(y == 1)
    index_2 = np.where(y == 2)
    # plot 4 images
    fig, sub = plt.subplots(2, 2)


    # use sklearn to get coefficients
    clf = SVC(kernel='linear', decision_function_shape='ovo')
    clf.fit(X, y)

    # plot region
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    sub[0][0].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    sub[0][0].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c='black', s=60, label='support vector')
    sub[0][0].scatter(X0[index_0], X1[index_0], c='r', s=50, marker='x', label='0')
    sub[0][0].scatter(X0[index_1], X1[index_1], c='g', s=50, marker='+', label='1')
    sub[0][0].scatter(X0[index_2], X1[index_2], c='b', s=50, marker='*', label='2')
    sub[0][0].legend()
    sub[0][0].set_xlim(xx.min(), xx.max())
    sub[0][0].set_ylim(yy.min(), yy.max())
    sub[0][0].set_xlabel('Sepal length')
    sub[0][0].set_ylabel('Sepal width')

    # poly kernel
    clf2 = SVC(kernel='poly', degree=2, decision_function_shape='ovo')
    clf2.fit(X, y)

    Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    sub[0][1].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    sub[0][1].scatter(clf2.support_vectors_[:, 0], clf2.support_vectors_[:, 1], c='black', s=60, label='support vector')
    sub[0][1].scatter(X0[index_0], X1[index_0], c='r', s=50, marker='x', label='0')
    sub[0][1].scatter(X0[index_1], X1[index_1], c='g', s=50, marker='+', label='1')
    sub[0][1].scatter(X0[index_2], X1[index_2], c='b', s=50, marker='*', label='2')
    sub[0][1].legend()
    sub[0][1].set_xlim(xx.min(), xx.max())
    sub[0][1].set_ylim(yy.min(), yy.max())
    sub[0][1].set_xlabel('Sepal length')
    sub[0][1].set_ylabel('Sepal width')

    # use LDA to reduce dimension, similar to PCA
    # calculate mean
    X = iris.data
    means = np.mean(X.reshape(-1, 50, 4), axis=1)
    overall_mean = np.mean(X, axis=0)

    # calculate S_w and S_b
    S_w = np.sum([(X[i * 50:(i + 1) * 50] - means[i]).T.dot(X[i * 50:(i + 1) * 50] - means[i]) for i in range(3)],
                 axis=0)
    S_b = 50 * (means - overall_mean).T.dot(means - overall_mean)

    # this part I ectually copy from my IML HW 2 PCA part
    mat = np.linalg.inv(S_w).dot(S_b)
    va, ve = np.linalg.eig(mat)
    eig_pairs = [(va[i], ve[:, i]) for i in range(len(va))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    w = np.vstack((eig_pairs[0][1], eig_pairs[1][1]))

    # reduce dimension of the data
    lda_X = X.dot(w.T)

    # re-train
    clf.fit(lda_X, y)

    X0, X1 = lda_X[:, 0], lda_X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    sub[1][0].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    sub[1][0].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c='black', s=60, label='support vector')
    sub[1][0].scatter(X0[index_0], X1[index_0], c='r', s=50, marker='x', label='0')
    sub[1][0].scatter(X0[index_1], X1[index_1], c='g', s=50, marker='+', label='1')
    sub[1][0].scatter(X0[index_2], X1[index_2], c='b', s=50, marker='*', label='2')
    sub[1][0].legend()
    sub[1][0].set_xlim(xx.min(), xx.max())
    sub[1][0].set_ylim(yy.min(), yy.max())
    sub[1][0].set_xlabel('Dimension 1 after LDA(2)')
    sub[1][0].set_ylabel('Dimension 2 after LDA(2)')

    # re-train polynomial model
    clf2.fit(lda_X, y)

    Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    sub[1][1].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    sub[1][1].scatter(clf2.support_vectors_[:, 0], clf2.support_vectors_[:, 1], c='black', s=60, label='support vector')
    sub[1][1].scatter(X0[index_0], X1[index_0], c='r', s=50, marker='x', label='0')
    sub[1][1].scatter(X0[index_1], X1[index_1], c='g', s=50, marker='+', label='1')
    sub[1][1].scatter(X0[index_2], X1[index_2], c='b', s=50, marker='*', label='2')
    sub[1][1].legend()
    sub[1][1].set_xlim(xx.min(), xx.max())
    sub[1][1].set_ylim(yy.min(), yy.max())
    sub[1][1].set_xlabel('Dimension 1 after LDA(2)')
    sub[1][1].set_ylabel('Dimension 2 after LDA(2)')
    plt.show()