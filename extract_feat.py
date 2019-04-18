from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
import os, pickle
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import VGG16, ResNet50
from dataset import Dataset, datagenerator
from keras import backend as K
from keras.optimizers import SGD
import cv2

PATH = 'input/train/'
dataset = Dataset()
x_train,x_val, labels = dataset.loadData()
def get_bgr_img(ids):
    ans = []
    colors = ['blue','green','red']
    flags = cv2.IMREAD_GRAYSCALE

    for id in ids:
        img = [cv2.imread(os.path.join(PATH, id+'_'+color+'.png'), flags).astype(np.float32)/255
               for color in colors]
        img = np.stack(img, axis=-1)
        img = cv2.resize(img, dsize=(226,226),interpolation=cv2.INTER_CUBIC)
        ans.append(img)
    return np.array(ans)
def get_y(labels, ids):
    ans = []
    for id in ids:
        label = labels.loc[id]['Target']
        ans.append(np.eye(28,dtype=np.float)[label].sum(axis=0))
    return np.array(ans)

def imageLoader(files, batch_size):
    L = len(files)
    #this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = get_bgr_img(files[batch_start:limit])
            Y = get_y(labels, files[batch_start:limit])
            print('AAAAAAAA')
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size
        print('BBBBBBB')


def main():
    vgg16 = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(226, 226, 3))
    print(vgg16.summary())
    nTrain = 30
    batch_size = 5

    train_features = np.zeros(shape=(nTrain, 7, 7, 512))
    train_labels = np.zeros(shape=(nTrain,28))


    model = keras.models.Sequential()
    model.add(vgg16)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu', input_dim=7 * 7 * 512))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(28, activation='sigmoid'))
    print(model.summary())
    #
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd)

    model.fit_generator(imageLoader(x_train[:30],batch_size), steps_per_epoch = 5, epochs=5)
    x_test = get_bgr_img(x_val)
    print('shape of x_test {}'.format(x_test.shape))
    preds = model.predict(x_test)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    print(preds)



    # print(x_train[:5])

    # x_train, x_val = datagenera   tor(x_train, x_val, img_rows=512, img_cols=512)
    # print(x_train.shape)
    # i = 0
    # while True:
    #     # print("AA")
    #     inputs_batch = np.array(get_bgr_img(PATH, x_train[i * batch_size : (i + 1) * batch_size]))
    #     # print(inputs_batch.shape)
    #     features_batch = vgg_conv.predict(inputs_batch)
    #     # print(features_batch.shape)
    #     train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    #     train_labels[i * batch_size : (i + 1) * batch_size] = get_y(labels,x_train[i * batch_size : (i + 1) * batch_size])
    #     i += 1
    #     if i * batch_size >= nTrain:
    #         break
    # pickle.dump((train_features,train_labels), open('feats_labels.pkl','wb'))
    # np.save('feat_label', (train_features,train_labels))
if __name__ == '__main__':
    main()
