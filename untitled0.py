# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:20:08 2024

@author: Eyal
"""
import torch
import torch.autograd
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.regression import MeanSquaredError
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import progressbar
import data_utils
import math
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os


class network(nn.Module):
    def __init__(self,width):
        super(network,self).__init__()
        #define layers
        self.float
        self.lstm1=nn.LSTM(2*width,1*width,batch_first=True)
        self.cnn1=nn.Conv2d(2, 10, 5,padding='same')
        self.tanh=nn.Tanh()
        #self.maxPool1=nn.MaxPool2d(5,stride=1,padding=2)
        self.cnn2=nn.Conv2d(10, 5, 5,padding='same')
        self.tanh2=nn.Tanh()
        #self.maxPool2=nn.MaxPool2d(3,stride=1,padding=1)
        self.cnn3=nn.Conv2d(5, 2, 5,padding='same')
        self.tanh3=nn.Tanh()
        #self.lstm2=nn.LSTM(1*width,2*width,batch_first=True)
        #self.lstm3=nn.LSTM(2*width,width,batch_first=True)
        #self.lin=nn.Linear(width, width)
    def forward(self,x):
        #forward pass
        p=x[0].reshape(x[0].shape[0],1,*x[0].shape[1:])
        x=torch.cat(x,dim=2)
        x=self.lstm1(x)[0]
        x=x.reshape(x.shape[0],1,*x.shape[1:])
        x=torch.cat((x,p),dim=1)
        x=self.cnn1(x)
        x=self.tanh(x)
        #x=self.maxPool1(x)
        x=self.cnn2(x)
        x=self.tanh2(x)
        #x=self.cnn3(x)
        #x=self.tanh3(x)
        x=torch.sum(x,dim=1,keepdim=False)
        #x=self.lstm2(x)[0]
        #x=self.lstm3(x)[0]
        #x=self.lin(x)
        return x

ind=0

load_test=data_utils.load_data_np("MatK","part_per_bin",r"C:\Users\Eyal\Documents\eyals junk\alpha\dataset_rot90_v3\train")
#load_test=data_utils.load_data_np("MatK","part_per_bin",r"C:\Users\Eyal\Documents\eyals junk\alpha\dataset_f\test")

"""for x in range(len(load_test[0])):
    load_test[0][x]=torch.rot90(load_test[0][x])
    load_test[1][x]=np.rot90(load_test[1][x])"""
#plt.imshow(load_test[1][0])
model=torch.load(r"C:\Users\Eyal\Documents\eyals junk\alpha\10_5_128/best_model")
model.eval()
y_exp=load_test[1][ind]
x2 = load_test[1][ind].copy()
x2[1:,:] = 0
x2=torch.from_numpy(x2)
x2=x2.type(torch.float32)
y_pred=model((load_test[0][ind].reshape([1,*load_test[0][ind].shape]),x2.reshape([1,*x2.shape]))).squeeze()
plt.imshow(load_test[0][ind])
plt.figure()
plt.imshow(y_exp)
plt.figure()
plt.imshow(y_pred.detach().numpy())
#plt.figure()
#plt.imshow(y_exp-y_pred.detach().numpy())
print(mean_squared_error(y_pred.detach().numpy(),y_exp))

mses=[]

#for x in range(len(load_test[0])):
for x in range(10):
    y_exp=load_test[1][x]
    x2 = load_test[1][x].copy()
    x2[1:,:] = 0
    x2=torch.from_numpy(x2)
    x2=x2.type(torch.float32)
    y_pred=model((load_test[0][x].reshape([1,*load_test[0][x].shape]),x2.reshape([1,*x2.shape]))).squeeze()

    mses.append(mean_squared_error(y_pred.detach().numpy(),y_exp))
print("\navg mse: "+str(sum(mses)/10))