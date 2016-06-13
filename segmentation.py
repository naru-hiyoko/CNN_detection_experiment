#//usr/local/bin/python2.7
# encoding: utf-8

import numpy as np
from numpy import float32
import sys
from os.path import join
import threading
from skimage.io import imshow, show, imread
from skimage.transform import resize, rescale
from scipy import ndimage as ndi
from skimage.io import imshow, show
import matplotlib.pyplot as plt
import chainer
from chainer import Function, FunctionSet, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


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
serializers.load_npz('./trained_30.model', model)

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

def heatmap(_title, _in):
    plt.imshow(_in, cmap='CMRmap')
    plt.axis('off')
    plt.title(_title)
    plt.show()
    #plt.savefig('')

def draw_heatmap3(A, B, C):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle(''.decode('utf-8'))
    
    ax1.imshow(A)
    ax1.axis('off')

    ax2.imshow(B, cmap='CMRmap')
    ax2.axis('off')
    
    ax3.imshow(C, cmap='CMRmap')
    ax3.axis('off')

    plt.show()
    
def threshold(_in, alpha):
    h, w = _in.shape
    output = _in.copy()
    thred = np.max(output) * alpha
    #thred = 0.80
    minval = np.min(output)
    maxval = np.max(output)
    for y in range(h):
        for x in range(w):
            if output[y, x] < thred:
                output[y, x] = False
            else:
                output[y, x] = True
                pass
    return output


def detection(image, alpha, beta):
    _in = image.transpose([2, 0, 1]).astype(float32)
    c, h, w = _in.shape
    _in = _in.reshape([1, c, h, w])
    _size = (h, w)
    outputA = forward_single(_in, _size)
    outputA_cleaned = threshold(outputA, alpha)
    label_objects, nb_labels = ndi.label(outputA_cleaned)
    sizes = np.bincount(label_objects.ravel())
    sizes[0] = 0
    mask_sizes = sizes > np.max(sizes) * beta
    cleaned = mask_sizes[label_objects]
    return cleaned

if __name__ == '__main__':
    image = imread(sys.argv[1])
    figure = image.copy()
    image = image.transpose([2, 0, 1]).astype(float32)
    c, h, w = image.shape
    image = image.reshape([1, c, h, w])
    _size = (h, w)

    alpha = 0.9
    beta = 0.9

    """ single scale """
    outputA = forward_single(image, _size)
    outputA_cleaned = threshold(outputA, alpha)
    
    """ multi scale integration """
    #outputB = forward_multi(image, _size)
    #outputB_cleaned = threshold(outputB, alpha)
    #Myutils.draw_heatmap3(figure / 255.0, outputA, outputB)

    """ labbeling algorithm applied. """
    label_objects, nb_labels = ndi.label(outputA_cleaned)
    sizes = np.bincount(label_objects.ravel())
    sizes[0] = 0
    mask_sizes = sizes > np.max(sizes) * beta
    cleaned = mask_sizes[label_objects]
    
    """ result is shown """
    draw_heatmap3(figure / 255.0, outputA, cleaned)    
