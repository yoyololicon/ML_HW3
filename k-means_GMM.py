import argparse
import Image
import numpy as np
from prettytable import PrettyTable
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from datetime import datetime

def log_likelihood(p):
    return np.sum(np.log(np.sum(p, axis=1)))

parser = argparse.ArgumentParser(description='ML HW3 - k-means & GMM')
parser.add_argument('img', help='your image file')
parser.add_argument('k', type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    img = Image.open(args.img)
    img.load()
    #img.show(title='origin')
    data = np.asarray(img, dtype='float')/255
    m, n, l = data.shape
    data = np.reshape(data, (-1, l))
    max_iter = 100
    k = args.k
    total_length = m * n

    #k-means part
    u = np.random.rand(k, l)
    r = np.full([total_length], k + 1)

    a = datetime.now().replace(microsecond=0)
    for i in range(300):
        dist = np.sum((data[:, None] - u) ** 2, axis=2)
        new_r = np.argmin(dist, axis=1)

        if np.array_equal(r, new_r):
            break
        else:
            r = new_r

        for j in range(k):
            data_k = data[np.where(r == j)]
            if len(data_k) == 0:
                u[j] = np.random.rand(l)
            else:
                u[j] = np.mean(data_k, axis=0)

    b = datetime.now().replace(microsecond=0)
    print '%d iterations for k = %d' % (i+1, k)
    print 'Time cost :', b-a
    table = PrettyTable()
    table.add_column("k-means mean value", range(k))
    table.add_column("r", np.round(u[:, 0]*255).astype('int'))
    table.add_column("g", np.round(u[:, 1]*255).astype('int'))
    table.add_column("b", np.round(u[:, 2]*255).astype('int'))
    print table

    new_data = np.round(u[r]*255)
    disp = Image.fromarray(new_data.reshape(m, n, l).astype('uint8'))
    disp.show(title='k-means')
    disp.save('k-means_'+str(k)+'.png')

    #GMM parts
    pi = np.array([len(np.where(r == i)[0])/float(total_length) for i in range(k)])
    cov = np.array([np.cov(data[np.where(r == i)].T) for i in range(k)])
    psb = np.array([multivariate_normal.pdf(data, mean=u[i], cov=cov[i]) for i in range(k)]).T * pi

    likelihood = []
    likelihood.append(log_likelihood(psb))

    a = datetime.now().replace(microsecond=0)
    for i in range(max_iter):
        #E step
        r = psb/np.sum(psb, axis=1)[:, None]

        #M step
        N = np.sum(r, axis=0)
        u = np.sum(data[:, None] * r[:, :, None], axis=0)/N[:, None]
        for j in range(k):
            cov[j] = ((data - u[j]) * r[:, j, None]).T.dot(data - u[j])/N[j]
        pi = N/total_length

        #evaluate
        for j in range(k):
            try:
                psb[:, j] = multivariate_normal.pdf(data, mean=u[j], cov=cov[j])*pi[j]
            except np.linalg.linalg.LinAlgError:
                print 'singular error at iteration', i + 1
                u[j] = np.random.rand(l)
                temp = np.random.rand(l, l)
                cov[j] = temp.dot(temp.T)
                psb[:, j] = multivariate_normal.pdf(data, mean=u[j], cov=cov[j])*pi[j]

        likelihood.append(log_likelihood(psb))

    b = datetime.now().replace(microsecond=0)
    print 'EM finished, time cost :', b-a

    plt.plot(likelihood)
    plt.title('GMM maximum likelihood curve')
    plt.xlabel('iterations')
    plt.ylabel('log p(x)', rotation=0)
    plt.show()

    table2 = PrettyTable()
    table2.add_column("GMM mean value", range(k))
    table2.add_column("r", np.round(u[:, 0]*255).astype('int'))
    table2.add_column("g", np.round(u[:, 1]*255).astype('int'))
    table2.add_column("b", np.round(u[:, 2]*255).astype('int'))
    print table2

    r = np.argmax(psb, axis=1)
    new_data = np.round(u[r]*255)
    disp2 = Image.fromarray(new_data.reshape(m, n, l).astype('uint8'))
    disp2.show(title='GMM')
    disp2.save('GMM_'+str(k)+'.png')