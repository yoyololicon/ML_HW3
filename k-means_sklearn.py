import argparse
import Image
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='ML HW3 - k-means')
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
    kmeans = KMeans(n_clusters=k, n_jobs=-1).fit(data)
    b = datetime.now().replace(microsecond=0)
    print 'Time cost :', b-a

    indice = kmeans.predict(data)
    new_data = np.round(kmeans.cluster_centers_[indice])
    disp = Image.fromarray(new_data.reshape(m, n, l).astype('uint8'))
    disp.show(title='k-means')

    table = PrettyTable()
    table.add_column("k-means mean value", range(k))
    table.add_column("r", np.round(kmeans.cluster_centers_[:, 0]).astype('int'))
    table.add_column("g", np.round(kmeans.cluster_centers_[:, 1]).astype('int'))
    table.add_column("b", np.round(kmeans.cluster_centers_[:, 2]).astype('int'))
    print table
