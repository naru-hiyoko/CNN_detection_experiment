# encoding: utf-8
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os
from os.path import join
from progressbar import ProgressBar, Bar, Percentage

""" ネガポジデータ読み込み pos:1, neg:0 """
def load_dataA():
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

def heatmap(_title, _in):
    plt.imshow(_in, cmap='CMRmap')
    plt.axis('off')
    plt.title(_title)
    plt.show()
    #plt.savefig('')

def normalMatrix(_title, _in, _row_labels, _col_labels):
    plt.imshow(_in, cmap='CMRmap')
    plt.title(_title)
    
    plt.set_xticks(np.arange(_in.shape[1]))
    plt.set_yticks(np.arange(_in.shape[0]))

    plt.invert_yaxis()
    plt.xaxis.tick_top()

    plt.set_xticklabels(_row_labels, minor=False)
    plt.set_yticklabels(_col_labels, minor=False)
    
    plt.show()
    
def draw_heatmap(data, row_labels, column_labels):
    # 描画する
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.show()
    plt.savefig('image.png')

    return heatmap

def draw_heatmap3(A, B, C):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle(''.decode('utf-8'))
    
    ax1.imshow(A)
    ax1.set_title('input'.decode('utf-8'))
    ax1.axis('off')

    ax2.imshow(B, cmap='CMRmap')
    ax2.set_title('single scale'.decode('utf-8'))
    ax2.axis('off')
    
    ax3.imshow(C, cmap='CMRmap')
    ax3.set_title('multi scale'.decode('utf-8'))
    ax3.axis('off')

    plt.show()
    
    

def threshold(_in):
    h, w = _in.shape
    output = _in.copy()
    thred = np.max(output) * 0.80
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


if __name__ == '__main__':
    load_dataA()
