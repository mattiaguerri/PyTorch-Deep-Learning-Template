import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from utils import device
from models import MyCNN, resnet18, Net1T


# # load val dataset
# val_ds = pd.read_pickle('./dataset/val/dfVal3.pkl')
# colsToExc = ['DATE', 'DeltaTimeTri', 'Lot']
# val_ds.drop(colsToExc, axis=1, inplace=True)
# tarCols = [x for x in val_ds.keys().to_list() if 'RsP' in x]
# dfFea = val_ds.drop(tarCols, axis=1)
# XVal = np.array(dfFea)
# yVal = np.array(val_ds[tarCols])

# load test dataset
test_ds = pd.read_pickle('./dataset/val/dfTest3.pkl')
colsToExc = ['DATE', 'DeltaTimeTri', 'Lot']
test_ds.drop(colsToExc, axis=1, inplace=True)
tarCols = [x for x in test_ds.keys().to_list() if 'RsP' in x]
dfFea = test_ds.drop(tarCols, axis=1)
XTest = np.array(dfFea)
yTest = np.array(test_ds[tarCols])

# load the model
inpSize=666; outSize=5; numHiddenLayers=3; hiddenSize=512; dropout=.0
model = Net1T(inpSize, outSize, numHiddenLayers, hiddenSize, dropout).to(device)
preTrainedModel = './checkpoint/1607450839.8576891-model.pt'
model.load_state_dict(torch.load(preTrainedModel))
model.eval()

# get predictions
inputs = torch.from_numpy(XTest).float().to(device)
pre = model.forward(inputs)
pre = pre.detach().cpu().numpy().squeeze()

# set the true y
tar = yTest

# compute relative error on the entire set
err = (np.abs(tar - pre) / tar).mean() * 100
print('\nError for the entire set =', err.round(2))

# try to reproduce MSELoss
mseLoss = ((tar-pre)**2).mean()
print('Try to reproduce MSELoss =', mseLoss)

# compute relative error in each range
print('\nError in each range:')
ranges = [[5.0, 6.5], [6.5, 7.5], [7.5, 10]]
for i in range(len(ranges)):
    ran = ranges[i]
    inds = np.where((tar>=ran[0]) & (tar<ran[1]))
    tarSel = tar[inds[0], inds[1]]
    preSel = pre[inds[0], inds[1]]
    err = (np.abs(tarSel - preSel) / tarSel).mean() * 100
    print('Range =', i+1, 'Error =', err.round(2))

# plot, only one target
tarInd = 4
print('\nPlot targets and predictions for', tarCols[tarInd])
tarSel = tar[:, tarInd]
preSel = pre[:, tarInd]
x = np.arange(tar.shape[0])
plt.figure(figsize=(20, 7))
plt.scatter(x, tarSel, c='k', label='Targets')
plt.scatter(x, preSel, c='r', label='Predictions')
plt.legend()
plt.show()