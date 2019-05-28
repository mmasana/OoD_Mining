import cPickle
import numpy as np
from scipy.misc import imresize
from skimage.io import imread


# Functions for loading CIFAR-10
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    if 'data' in dict:
        dict['data'] = dict['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2) / 256.
    return dict


def load_cifar10_one(f):
    batch = unpickle(f)
    print("Loading %s: %d" % (f, len(batch['data'])))
    return batch['data'], batch['labels']


def load_cifar10_set(files, data_dir):
    data, labels = load_cifar10_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_cifar10_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    return data, labels


def load_cifar10(data_dir):
    train_files = ['data_batch_%d' % d for d in xrange(1, 6)]
    c10_trn_x, c10_trn_y = load_cifar10_set(train_files, data_dir)
    pi = np.random.permutation(len(c10_trn_x))
    c10_trn_x, c10_trn_y = c10_trn_x[pi], c10_trn_y[pi]
    c10_tst_x, c10_tst_y = load_cifar10_set(['test_batch'], data_dir)
    c10_tst_y = np.asarray(c10_tst_y)
    return c10_trn_x, c10_trn_y, c10_tst_x, c10_tst_y


# Function for loading the other datatsets using a txt file with the list of images (with PATH if needed) to load
def load_test(path, inp_size):
    fid = open(path + "images.txt", "r")
    img_names = fid.read().splitlines()
    fid.close()
    tst_img = np.zeros([len(img_names), inp_size, inp_size, 3])
    for m in xrange(len(img_names)):
        data = imread(path + "images/" + img_names[m])
        if len(data.shape) == 2:
            data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        tst_img[m, :, :, :] = imresize(data, (inp_size, inp_size, 3)) / 255.0

    return tst_img
