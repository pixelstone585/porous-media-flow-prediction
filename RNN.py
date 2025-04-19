
# -*- coding: utf-8 -*-
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
import random
# declaring log variable
log=""
run_name=input("enter identifier: ")
os.mkdir(run_name)
file=open(run_name+r"/log.txt","x")
file.close()
def save_log(log):
    file=open(run_name+r"/log.txt","w")
    file.write("\n---------------------------\n"+log)
    file.close()
def shuffle_2_lists(list1,list2):
    listonumbers=[]
    for x in range(len(list1)):
        listonumbers.append(x)
    random.shuffle(listonumbers)
    list1 = [list1[i] for i in listonumbers]
    list2 = [list2[i] for i in listonumbers]
    return(list1,list2)
try:
    #load dataset
    #tmp=data_utils.load_data_sliced("MatK","part_per_bin",r"C:\Users\Eyal\Documents\eyals junk\alpha\data",step_size=40)
    #for generating new dataset V
    #data_utils.generate_dataset("MatK","part_per_bin",r"C:\Users\Eyal\Documents\eyals junk\alpha\Var 1",r"C:\Users\Eyal\Documents\eyals junk\alpha\dataset",step_size=40)"
    load_train=data_utils.load_data_np("MatK","part_per_bin",r"C:\Users\Eyal\Documents\eyals junk\alpha\dataset_rot90_s\train")
    load_train=shuffle_2_lists(*load_train)
    load_test=data_utils.load_data_np("MatK","part_per_bin",r"C:\Users\Eyal\Documents\eyals junk\alpha\dataset_rot90_s\test")
except Exception as e:
    log+="["+str(datetime.datetime.now())+"]"+"ERROR while loading data:"+str(e)+"\n"
    save_log(log)
    raise e
log+="["+str(datetime.datetime.now())+"]"+"loaded data with no errors"+"\n"
#add stuff here
#x_train,x_test,y_train,y_test=train_test_split(tmp[0],tmp[1],test_size=0.33,random_state=42)
tmp={"X_train":load_train[0],"y_train":load_train[1],"X_test":load_test[0],"y_test":load_test[1]}
class f____youpytorch(Dataset):
    def __init__(self):
        "useless function"
    def __getitem__(self, index):
        x2 = tmp["y_train"][index].copy()
        x2[1:,:] = 0
        x2=torch.from_numpy(x2)
        x2=x2.type(torch.float32)
        return ((tmp["X_train"][index],x2), torch.from_numpy(tmp["y_train"][index]))

    def __len__(self):
       return len(tmp["X_train"])

#df=pd.DataFrame.from_dict(tmp)
#X_train, X_test, y_train, y_test = train_test_split(tmp["X"], tmp["Y"], test_size=0.20,random_state=7)
#define network


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

def train(model,bar,batch_size,epoch=500):
    #train model
    best_score=1e9
    loader=torch.utils.data.DataLoader(f____youpytorch(),batch_size=batch_size,num_workers=0)
    losses = []
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    counter=0
    global log
    for e in range(epoch):
        counter+=1
        bar.update(counter)
        for i, data in enumerate(loader):
            #process batch
            #y_expec=torch.from_numpy(tmp["y_train"][s]).reshape([1,*tmp["y_train"][s].shape])
            #X_study=tmp["X_train"][s]
            X_study, y_expec = data
            optimizer.zero_grad()
            y_hat=model.forward(X_study)
            y_hat=y_hat.type(torch.float64)
            loss = loss_func(y_hat, y_expec)
            #warn if network is outputting nans
            if True in torch.isnan(y_hat):
                log+="["+str(datetime.datetime.now())+"]"+"WARNING detected nan in model prediction\n"
            #------------------------------
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
        if loss.detach().item() <= best_score:
            best_score = loss.detach().item()
            torch.save(model,run_name+r"/best_model")
        losses+=[loss.detach().item()]
        torch.save(model,run_name+r"/last_model")
        log+="["+str(datetime.datetime.now())+"]"+"epoch "+str(e)+" completed\n"
        log+="["+str(datetime.datetime.now())+"]"+"loss "+str(loss)+"\n"
        save_log(log)
    return counter,losses;
def train_full(tgt_epoch,batch_size):
    #add qol features to training process
    model=network(120)
    bar_widgets=[progressbar.FormatLabel("training model")," ",progressbar.Percentage(), " ", progressbar.GranularBar(markers=" ▁▂▃▄▅▆▇█")," ",progressbar.AdaptiveETA(),]
    bar_max=tgt_epoch
    bar = progressbar.ProgressBar(max_value=bar_max,widgets=bar_widgets)
    bar.update(0)
    counter,losses=train(model,bar,batch_size,tgt_epoch)
    fig=plt.figure()
    listonumbers=[]
    
    for x in range(len(losses)):
        listonumbers+=[x+1]
    plt.plot(listonumbers,np.log10(losses))
    plt.savefig(run_name+r"/loss graph.png")
    f=open(run_name+r"/losses.txt","x")
    f.write(str(losses))
    f.close()
    torch.save(model,run_name+r"/last_model")
    print("\ndone!")
    return counter
#starts training
error_sample_size=len(tmp["X_test"])
log+="["+str(datetime.datetime.now())+"]"+"training started\n"
try:
    counter=train_full(300,128)
    "dummy statement"
except Exception as e:
    log+="["+str(datetime.datetime.now())+"]"+"ERROR while training: "+str(e)+"\n"
    save_log(log)
    raise e
log+="["+str(datetime.datetime.now())+"]"+"training completed with no errors\n"

model=torch.load(run_name+r"/best_model")
#calculate avg error for 10 random test samples

#cosine_sim=CosineSimilarity(reduction = 'mean')
mses=[]
try:
    model.eval()
    for x in range(error_sample_size):
        y_exp=tmp["y_test"][x]
    
        y_pred=model(tmp["X_test"][x].reshape([1,*tmp["X_test"][x].shape])).squeeze()
        #plt.imshow(y_exp)
        #plt.imshow(y_pred.deatch().numpy())
        #plt.imshow(y_exp-y_pred.deatch().numpy())
        mses.append(mean_squared_error(y_pred.detach().numpy(),y_exp))
    print("\navg mse: "+str(sum(mses)/error_sample_size))
    log+="["+str(datetime.datetime.now())+"]"+"model evaluation complete, score: "+str(sum(mses)/error_sample_size)+"\n"
except Exception as e:
    log+="["+str(datetime.datetime.now())+"]"+"ERROR while evaluating model: "+str(e)+"\n"
    save_log(log)
    raise e
save_log(log)