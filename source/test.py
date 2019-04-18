import numpy as np
import pandas as pd
import cv2, os
from sklearn.metrics import f1_score
import scipy.optimize as opt

PATH = './'
TRAIN = '../input/train/'
TEST = '../input/test/'
LABELS = '../input/train.csv'
SAMPLE = '../input/sample_submission.csv'

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

labels = pd.read_csv(LABELS).set_index('Id')
submit = pd.read_csv(SAMPLE).set_index('Id')
train_names = labels.index.values
test_names = submit.index.values

def F1_soft(preds,targs,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)
    return score

def fit_val(x,y):
    params = 0.5*np.ones(len(name_label_dict))
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x,y,p) - 1.0,
                                      wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p

def save_pred(pred, th=0.5, fname='my_protein_classification.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line>th)[0]]))
        pred_list.append(s)
    sample_df = pd.read_csv(SAMPLE)
    sample_list = list(sample_df.Id)
    pred_dic = dict((key, value) for (key, value)
                in zip(test_names,pred_list))
    pred_list_cor = [pred_dic[id] for id in sample_list]
    df = pd.DataFrame({'Id':sample_list,'Predicted':pred_list_cor})
    df.to_csv(fname, header=True, index=False)

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

y_resnet_v = np.load('../pred/resnet34_256_y_v.npy')

# pred_vgg_bgry_v = np.load('../pred/vgg16_224_BRGY_preds_val.npy')
pred_vgg_bgry_v = np.load('../pred/vgg16_224_BRGY_preds_val.npy')

pred_resnet_bgry_v = np.load('../pred/resnet_256_BRGY_preds_val.npy')
# pred_resnet_bgry_v = np.load('../pred/resnet_256_BRGY_preds_val.npy')

pred_resnet_t = np.load('../pred/resnet34_256_preds_t.npy')
pred_resnet_v = np.load('../pred/resnet34_256_preds_v.npy')

pred_vgg_t = np.load('../pred/vgg16_preds_t.npy')
pred_vgg_v = np.load('../pred/vgg16_preds_v.npy')

pred_incep_t = np.load('../pred/incept_preds_t_224.npy')
pred_incep_v = np.load('../pred/incept_preds_v_224.npy')

pred_green_t = np.load('../pred/vgg16_224_green_preds_test.npy')
pred_green_v = np.load('../pred/vgg16_224_green_preds_val.npy')

pred_blue_t = np.load('../pred/vgg16_224_blue_preds_test.npy')
pred_blue_v = np.load('../pred/vgg16_224_blue_preds_val.npy')

pred_red_t = np.load('../pred/vgg16_224_red_preds_test.npy')
pred_red_v = np.load('../pred/vgg16_224_red_preds_val.npy')

pred_yel_t = np.load('../pred/vgg16_224_yel_preds_test.npy')
pred_yel_v = np.load('../pred/vgg16_224_yel_preds_val.npy')
# y_t = np.load('../pred/vgg_y_v.npy')

th_t = np.array([0.565,0.39,0.55,0.345,0.33,0.39,0.33,0.45,0.38,0.39,
               0.34,0.42,0.31,0.38,0.49,0.50,0.38,0.43,0.46,0.40,
               0.39,0.505,0.37,0.47,0.41,0.545,0.32,0.1])

pred_v = (pred_resnet_v*3+pred_vgg_v*7)/10
# pred_v = (pred_incep_v*0 + pred_resnet_v*3 + pred_vgg_v*7)/10
# pred_t = (pred_incep_t*2 + pred_resnet_t*2 + pred_vgg_t*6)/10
# pred_v = pred_green_v + pred_red_v + pred_blue_v + pred_yel_v
# pred_v = sigmoid_np(pred_v)
pred_t = pred_incep_t
print(pred_t[0])
# y_v = y_resnet_v
# th = fit_val(pred_v,y_v)
# th[th<0.1] = 0.1
# print('Thresholds: ',th)
# print('F1 macro: (th = th_t)',f1_score(y_v, pred_v>th_t, average='macro'))
# print('F1 macro (th = 0.5): ',f1_score(y_v, pred_v>0.5, average='macro'))
# print('F1 macro (th = th): ',f1_score(y_v, pred_v>th, average='macro'))

#
# save_pred(pred_t,0.5,'danh__INCEPT_05.csv')
# save_pred(pred_t,th_t,'danh__INCEPT_thmaf.csv')
