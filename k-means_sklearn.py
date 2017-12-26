import argparse
import Image
import numpy as np
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='ML HW3 - k-means & GMM')
parser.add_argument('img', help='hw3_img.jpg')

if __name__ == '__main__':
    args = parser.parse_args()
    img = Image.open(args.img)
    img.load()
    img.show(title='origin')
    data = np.asarray(img, dtype='float')/255
    m, n = data.shape[:2]
    data = np.reshape(data, (-1, 3))

    for k in (2, 3, 5, 20):
        kmeans = KMeans(n_clusters=k, n_jobs=-1).fit(data)
        indice = kmeans.predict(data)
        new_data = kmeans.cluster_centers_[indice] * 255
        disp = Image.fromarray(new_data.reshape(m, n, 3).astype('uint8'))
        disp.show(title='k = %d' % k)
