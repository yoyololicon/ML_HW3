import argparse
import Image
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='ML HW3 - k-means & GMM')
parser.add_argument('img', help='hw3_img.jpg')
parser.add_argument('k', type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    img = Image.open(args.img)
    img.load()
    img.show(title='origin')
    data = np.asarray(img, dtype='float')/255
    m, n, l = data.shape
    data = np.reshape(data, (-1, l))
    max_iter = 100
    k = args.k

    gmm = GMM(n_components=k).fit(data)
    indice = gmm.predict(data)
    new_data = gmm.means_[indice] * 255
    disp = Image.fromarray(new_data.reshape(m, n, l).astype('uint8'))
    disp.show(title='k = %d' % k)

    table = PrettyTable()
    table.add_column("mean number", range(k))
    table.add_column("r", (gmm.means_[:, 0] * 255).astype('int'))
    table.add_column("g", (gmm.means_[:, 1] * 255).astype('int'))
    table.add_column("b", (gmm.means_[:, 2] * 255).astype('int'))
    print table
