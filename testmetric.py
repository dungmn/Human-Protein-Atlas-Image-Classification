import numpy as np
def acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()

preds = np.array([2.7, 1.6, 3.5,1.9,0.5])
y = np.array([1.7, 1.8, 3.0,2,1.5])
print(acc(preds,y,0.1))
