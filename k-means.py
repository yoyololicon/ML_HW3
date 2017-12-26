import argparse
import Image
import numpy as np
from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='ML HW3 - k-means & GMM')
parser.add_argument('img', help='hw3_img.jpg')

if __name__ == '__main__':
    args = parser.parse_args()
    img = Image.open(args.img)
    img.load()
    img.show(title='origin')
    data = np.asarray(img, dtype='float')/255
    m, n, l = data.shape
    data = np.reshape(data, (-1, l))
    max_iter = 300

    for k in (2, 3, 5, 20)[:3]:
        u = np.random.rand(k, l)
        r = np.full([len(data)], k+1)

        for j in range(max_iter):
            dist = np.sum((data[:, None] - u)**2, axis=2)
            new_r = np.argmin(dist, axis=1)

            if np.array_equal(r, new_r):
                break
            else:
                r = new_r

            for i in range(k):
                data_k = data[np.where(r == i)]
                if len(data_k) == 0:
                    u[i] = np.random.rand(l)
                else:
                    u[i] = np.mean(data_k, axis=0)

        print '%d iterations for k = %d' % (j, k)
        table = PrettyTable(["r", "g", "b"])
        for j in u:
            table.add_row((j * 255).astype('int'))
        print table
        new_data = u[r] * 255
        disp = Image.fromarray(new_data.reshape(m, n, l).astype('uint8'))
        disp.show(title='k = %d' % k)
