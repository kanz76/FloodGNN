import numpy as np
from sklearn import metrics 



def compute_MSE_per_step(targets, preds):
    mse =  np.mean((targets - preds) ** 2, axis=0)
    return mse 

def compute_r2_metrics(targets, preds):
    r2 = metrics.r2_score(targets, preds, multioutput='raw_values')
    return r2
