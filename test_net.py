#//usr/local/bin/python2.7
# encoding: utf-8

import numpy as np
from numpy import float32
import sys
from os.path import join
from progressbar import ProgressBar, Percentage, Bar
import logging
import threading

from skimage.io import imshow, show, imread
from skimage.transform import resize, rescale
from scipy import ndimage as ndi
from skimage.io import imshow, show

import chainer
from chainer import Function, FunctionSet, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import Myutils

""" network definition """

model = FunctionSet(
    conv1 = F.Convolution2D(3, 32, 5, stride=1, pad=2),
    norm1 = F.BatchNormalization(32),
    conv2 = F.Convolution2D(32, 32, 5, stride=1, pad=2),
    norm2 = F.BatchNormalization(32),
    conv3 = F.Convolution2D(32, 16, 5, stride=1, pad=2),
    norm3 = F.BatchNormalization(16),
    conv4 = F.Convolution2D(16, 2, 5, stride=1, pad=0),
    ip1 = F.Linear(2, 2)
)


""" load trained model """
serializers.load_npz('../data/snapshot/trained_30.model', model)

def forward_single(x_data, _size, train=False):
    datum = x_data[0].transpose([1, 2, 0]) / 255.0
    datum = datum.transpose([2, 0, 1])
    c, h, w = datum.shape
    datum = datum.reshape([1, c, h, w])
    x = Variable(datum)

    
    h = model.conv1(x)
    h = model.norm1(h)
    h = F.relu(h)
    h = F.max_pooling_2d(h, 3, stride=2)

    h = model.conv2(h)
    h = model.norm2(h)
    h = F.relu(h)
    h = F.max_pooling_2d(h, 3, stride=2)

    h = model.conv3(h)
    h = model.norm3(h)
    h = F.relu(h)
    h = F.average_pooling_2d(h, 3, stride=2)
    
    h = model.conv4(h)

    h = F.softmax(h)
    y = h.data

    """ positive 領域 """
    fmap = resize(y[0][1], _size).astype(np.float32)
    return fmap


def forward_multi(x_data, _size, train=False):
    scale = [0.7, 0.85, 1.00, 1.25, 1.5]
    global_output = []

    #for s in scale:
    def compute(s):

        datum = x_data[0].transpose([1, 2, 0]) / 255.0
        datum = rescale(datum, s).astype(np.float32)
        datum = datum.transpose([2, 0, 1])
        
        c, h, w = datum.shape
        datum = datum.reshape([1, c, h, w])

        x = Variable(datum)

        h = model.conv1(x)
        h = model.norm1(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)

        h = model.conv2(h)
        h = model.norm2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)

        h = model.conv3(h)
        h = model.norm3(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 3, stride=2)
    
        h = model.conv4(h)
    
        h = F.softmax(h)
        y = h.data

        """ positive 領域 """
        fmap = resize(y[0][1], _size).astype(np.float32)
        global_output.append(fmap)

    threads = []
    for s in scale:
        th = threading.Thread(target=compute, args=[s])
        th.start()
        threads.append(th)

    for i in range(5):
        threads[i].join()

    
    global_output = np.asarray(global_output)
    return np.max(global_output, axis=0)


""" testing """


if __name__ == '__main__':
    image = imread(sys.argv[1])
    figure = image.copy()
    image = image.transpose([2, 0, 1]).astype(float32)
    c, h, w = image.shape
    image = image.reshape([1, c, h, w])
    _size = (h, w)

    outputA = forward_single(image, _size)
    outputA_cleaned = Myutils.threshold(outputA)

    #outputB = forward_multi(image, _size)
    #outputB_cleaned = Myutils.threshold(outputB)

    #Myutils.draw_heatmap3(figure / 255.0, outputA, outputB)

    label_objects, nb_labels = ndi.label(outputA)
    sizes = np.bincount(label_objects.ravel())
    sizes[0] = 0
    mask_sizes = sizes > np.max(sizes) * 0.9
    cleaned = mask_sizes[label_objects]
    
    Myutils.draw_heatmap3(figure / 255.0, outputA, cleaned)    
