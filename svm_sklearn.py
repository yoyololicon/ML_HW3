import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

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
    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target

    clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
    #clf = svm.LinearSVC()
    clf.fit(X, y)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    index_0 = np.where(y == 0)
    index_1 = np.where(y == 1)
    index_2 = np.where(y == 2)
    supvec = clf.support_vectors_
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    plt.scatter(supvec[:, 0],supvec[:, 1], c='black', s=60)
    plt.scatter(X0[index_0], X1[index_0], c='r', s=50, marker='x')
    plt.scatter(X0[index_1], X1[index_1], c='g', s=50, marker='+')
    plt.scatter(X0[index_2], X1[index_2], c='b', s=50, marker='*')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xticks(())
    plt.yticks(())
    plt.show()