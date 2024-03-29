from fastai.conv_learner import *
from fastai.dataset import *
import pandas as pd
import numpy as np
import os, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.optimize as opt

PATH = './'
TRAIN = './input/train/'
TEST = './input/test/'
LABELS = './input/train.csv'
SAMPLE = './input/sample_submission.csv'

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

nw = 2   #number of workers for data loader
arch = vgg16 #specify target architecture
arch = resnet34 #specify target architecture

labels = pd.read_csv(LABELS).set_index('Id')
submit = pd.read_csv(SAMPLE).set_index('Id')
train_names = labels.index.values
print(len(train_names))
test_names = submit.index.values
tr_n, val_n = train_test_split(train_names, test_size=0.1, random_state=42)
print(len(tr_n),len(val_n))

def open_rgby(path,id): #a function that reads RGBY image
    colors = ['red','green','blue']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id+'_'+color+'.png'), flags).astype(np.float32)/255
           for color in colors]
    return np.stack(img, axis=-1)

class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.labels = pd.read_csv(LABELS).set_index('Id')
        self.labels['Target'] = [[int(i) for i in s.split()] for s in self.labels['Target']]
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        img = open_rgby(self.path,self.fnames[i])
        if self.sz == 512: return img
        else: return cv2.resize(img, (self.sz, self.sz),cv2.INTER_AREA)

    def get_y(self, i):
        if(self.path == TEST): return np.zeros(len(name_label_dict),dtype=np.int)
        else:
            labels = self.labels.loc[self.fnames[i]]['Target']
            return np.eye(len(name_label_dict),dtype=np.float)[labels].sum(axis=0)

    @property
    def is_multi(self): return True
    @property
    def is_reg(self):return True
    #this flag is set to remove the output sigmoid that allows log(sigmoid) optimization
    #of the numerical stability of the loss function

    def get_c(self): return len(name_label_dict) #number of classes

def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(30, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    #mean and std in of each channel in the train set
    stats = A([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313])
    tfms = tfms_from_stats(stats, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
                aug_tfms=aug_tfms)
    ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],TRAIN),
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    print('Danh dmmmm')
    print(md)
    return md
print('AAAAA')
get_data(sz, bs)
def display_imgs(x):
    columns = 4
    bs = x.shape[0]
    rows = min((bs+3)//4,4)
    fig=plt.figure(figsize=(columns*4, rows*4))
    for i in range(rows):
        for j in range(columns):
            idx = i+j*columns
            fig.add_subplot(rows, columns, idx+1)
            plt.axis('off')
            plt.imshow((x[idx,:,:,:3]*255).astype(np.int))
    plt.show()

# display_imgs(np.asarray(md.trn_ds.denorm(x)))

def find_mean_std_train(md):
    x_tot = np.zeros(4)
    x2_tot = np.zeros(4)
    ii=0
    for x,y in iter(md.trn_dl):
        ii = ii + 1
        x = md.trn_ds.denorm(x).reshape(-1,4)
        x_tot += x.mean(axis=0)
        x2_tot += (x**2).mean(axis=0)

    channel_avr = x_tot/len(md.trn_dl)
    channel_std = np.sqrt(x2_tot/len(md.trn_dl) - channel_avr**2)
    return channel_avr,channel_std

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()

def acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()

sz = 256 #image size
bs = 64  #batch size

md = get_data(sz,bs)

# print(len(md.val_ds))
print('start training ...')
st = time.time()
learner = ConvLearner.pretrained(arch,md,ps=0.5) #dropout 50%
print(learner.summary())
learner.clip = 1.0 #gradient clipping
learner.opt_fn = optim.Adam
learner.crit = FocalLoss()
learner.metrics = [acc]
lr = 2e-2
# lr = learner.lr_find()

learner.fit(lr,1)
learner.unfreeze()
lrs=np.array([lr/10,lr/3,lr])
learner.fit(lrs/4,4,cycle_len=2,use_clr=(10,20))
learner.fit(lrs/4,2,cycle_len=4,use_clr=(10,20))
learner.fit(lrs/16,1,cycle_len=8,use_clr=(5,20))
# learner.save('My_ResNet34_256_1')
learner.load('My_ResNet34_256_1')
print('done load model')

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

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
                in zip(learner.data.test_ds.fnames,pred_list))
    pred_list_cor = [pred_dic[id] for id in sample_list]
    df = pd.DataFrame({'Id':sample_list,'Predicted':pred_list_cor})
    df.to_csv(fname, header=True, index=False)
#validation
# preds,y = learner.TTA(n_aug=16)
# preds = np.stack(preds, axis=-1)
# preds = sigmoid_np(preds)
# pred = preds.max(axis=-1)
# print(pred)

# th = fit_val(pred,y)
# th[th<0.1] = 0.1
# print('Thresholds: ',th)
# print('F1 macro: ',f1_score(y, pred>th, average='macro'))
# print('F1 macro (th = 0.5): ',f1_score(y, pred>0.5, average='macro'))
# print('F1 micro: ',f1_score(y, pred>th, average='micro'))
#
# print('Fractions: ',(pred > th).mean(axis=0))
# print('Fractions (true): ',(y > th).mean(axis=0))


th_t = np.array([0.565,0.39,0.55,0.345,0.33,0.39,0.33,0.45,0.38,0.39,
               0.34,0.42,0.31,0.38,0.49,0.50,0.38,0.43,0.46,0.40,
               0.39,0.505,0.37,0.47,0.41,0.545,0.32,0.1])


# preds_t,y_t = learner.TTA(n_aug=16,is_test=True)
# preds_t = np.stack(preds_t, axis=-1)
# preds_t = sigmoid_np(preds_t)
# pred_t = preds_t.max(axis=-1) #max works better for F1 macro score
# print(pred_t)
# print('Fractions: ',(pred_t > th_t).mean(axis=0))

# save_pred(pred_t,th_t)
print("Time: {}".format(time.time()-st))
