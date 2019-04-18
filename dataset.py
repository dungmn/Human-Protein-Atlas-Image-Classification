# from fastai.conv_learner import *
# from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.optimize as opt
import cv2
from keras import backend as K

class Dataset():
    """docstring for Dataset."""
    def __init__(self, path_train = 'input/train/', path_test='input/test',path_labels='input/train.csv'):
        self.path_train = path_train
        self.path_labels = path_labels

        # self.path_test = path_test

    def loadData(self, testsize = 0.1):
        labels = pd.read_csv(self.path_labels).set_index('Id')
        labels['Target'] = [[int(i) for i in s.split()] for s in labels['Target']]
        train_names = labels.index.values
        # print(len(train_names))
        # train_np = []
        # for id in train_names:
        #     train_np.append(get_bgr_img(self.path_train,id))
        # # test_names = list({f[:36] for f in os.listdir(self.path_test)})
        train_n, val_n = train_test_split(train_names, test_size=testsize, random_state=42)
        return train_n, val_n, labels

    #a function that reads RGBY image



# PATH = './'
# TRAIN = '../input/train/'
# TEST = '../input/test/'
# LABELS = '../input/train.csv'
# SAMPLE = '../input/sample_submission.csv'

name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',
14:  'Microtubules',
15:  'Microtubule ends',
16:  'Cytokinetic bridge',
17:  'Mitotic spindle',
18:  'Microtubule organizing center',
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',
22:  'Cell junctions',
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',
27:  'Rods & rings' }

def datagenerator(x_train,img_rows,img_cols):
    if K.image_data_format() == 'channels_first':
        print("AAA")
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        print("BBB")
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        # print(x_train.shape)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        # print(input_shape)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return x_train, x_test
# nw = 2   #number of workers for data loader
# arch = resnet34 #specify target architecture
labels = pd.read_csv('input/train.csv').set_index('Id')
labels['Target'] = [[int(i) for i in s.split()] for s in labels['Target']]
a = labels.index.values
l = labels.loc[a[2:5]]['Target']
