import argparse
import Image
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from prettytable import PrettyTable
from datetime import datetime

parser = argparse.ArgumentParser(description='ML HW3 - GMM')
parser.add_argument('img', help='hw3_img.jpg')
parser.add_argument('k', type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    img = Image.open(args.img)
    img.load()
    #img.show(title='origin')
    data = np.asarray(img, dtype='float')
    m, n, l = data.shape
    data = np.reshape(data, (-1, l))
    k = args.k

    a = datetime.now().replace(microsecond=0)
    gmm = GMM(n_components=k).fit(data)
    b = datetime.now().replace(microsecond=0)
    print 'Time cost :', b-a

    indice = gmm.predict(data)
    new_data = np.round(gmm.means_[indice])
    disp = Image.fromarray(new_data.reshape(m, n, l).astype('uint8'))
    disp.show(title='GMM')

    table = PrettyTable()
    table.add_column("GMM mean value", range(k))
    table.add_column("r", np.round(gmm.means_[:, 0]).astype('int'))
    table.add_column("g", np.round(gmm.means_[:, 1]).astype('int'))
    table.add_column("b", np.round(gmm.means_[:, 2]).astype('int'))
    print table
    print gmm.n_iter_, 'of iterations'
    print 'log likelihood', gmm.lower_bound_
