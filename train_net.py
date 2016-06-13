# encoding: utf-8

import numpy as np
import sys
from os.path import join
from progressbar import ProgressBar, Percentage, Bar
import logging

import cPickle
import chainer
from chainer import cuda
from chainer import Function, FunctionSet, Variable, optimizers, serializers, gradient_check, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

""" ネガポジデータ読み込み pos:1, neg:0 """
def load_data():
    prefix = '../data/pkl'
    data = []
    label = np.asarray([], dtype=np.uint8)

    prog = ProgressBar()
    prog.max_value = len(os.listdir(prefix)) - 1

    t = 0
    for file in os.listdir(prefix):
        prog.update(t)
        t = t+1
        
        if os.path.exists(join(prefix, file)) and '.pkl' in file:
            with open(join(prefix, file), 'rb') as f:
                data_dic = cPickle.load(f)
                data.append(data_dic['data'])
                label = np.append(label, data_dic['labels'])
    data = np.vstack(data)
    return data, label


train_data, train_labels = load_data()
train_labels = train_labels.astype(np.int32)

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

def forward(x_data, y_data, train=True):
    x, t = Variable(cuda.to_gpu(x_data)), chainer.Variable(cuda.to_gpu(y_data))
    
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

    h = model.ip1(h)
    y = h

    """ softmax normalization + compute loss & validation """
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

""" training configuration """
iteration = 50
batchsize = 100
N = train_data.shape[0]
model.to_gpu()
optim = optimizers.Adam()
optim.setup(model)
logging.basicConfig(filename='train.log', filemode='w', level=logging.DEBUG)

""" training """

for epoch in range(iteration):
    perm = np.random.permutation(N)
    progress = ProgressBar(widgets=['epoch: '+str(epoch+1)+' ',Percentage()])
    progress.min_value = 0
    progress.max_value = N-1
    progress.start()
    sum_loss = 0.0
    sum_accuracy = 0.0
    
    for i in range(0, N, batchsize):
        
        data_batch = train_data[perm[i:i+batchsize]]
        label_batch = train_labels[perm[i:i+batchsize]]
        optim.zero_grads()
        loss, accuracy = forward(data_batch, label_batch, train=True)
        loss.backward()
        optim.update()

        sum_loss += loss.data * batchsize
        sum_accuracy += accuracy.data * batchsize

        status = 'epoch: {} , loss: {:0.4f}, acc: {:0.4f}'.format(epoch, float(loss.data), float(accuracy.data))
        progress.widgets = [status, Percentage()]
        progress.update(i)
        
    print (' [epoch : %d ,  loss : %f , accuracy : %f]' % (epoch+1, sum_loss / N, sum_accuracy / N))
    logging.info(' [epoch : %d ,  loss : %f , accuracy : %f]' % (epoch+1, sum_loss / N, sum_accuracy / N))

    prefix = '../data/snapshot'
    serializers.save_npz(join(prefix, 'trained_%d.model' % (epoch+1)), model)
    serializers.save_npz(join(prefix, 'state_%d.state' % (epoch+1)), optim)
