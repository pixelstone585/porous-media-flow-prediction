# -*- coding: utf-8 -*-
import torch
#import torchvision.transforms as transforms
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import progressbar
import math
from PIL import Image as im 
import os 
import shutil
import random
from sklearn.model_selection import train_test_split
# load generated data
# def load_dataset(x_dir,y_dir):
#     #get list of files to load
#     files_x=os.listdir(x_dir)
#     #define variables
#     #to_tensor = transforms.Compose([transforms.ToTensor()])
#     data_x=[]
#     data_y=[]
#     for file in files_x:
#         #load fiels and store with label
#         data_x+=[to_tensor(im.open(x_dir+"/"+file))]
#         data_y+=[np.load(y_dir+"/"+(file.replace(".png",".npy").replace("_in_","_out_")))]
#     return(data_x,data_y)
def get_subarrays2(array,size_0, size_1, step_size_0, step_size_1,end_index_1="placeholder", start_index_1=0):
    slices = []
    if end_index_1=="placeholder":
        end_index_1 = array.shape[1]
    for n0 in range(0, array.shape[0]-size_0, step_size_0):
        for n1 in range(start_index_1, end_index_1-size_1, step_size_1):
            slices.append( array[n0:n0+size_0, n1:n1+size_1] )
    return slices

def get_a_subarray(array,size):
    x=array.shape[0]-size
    y=array.shape[1]-size
    slices=[]
    slices=array[x:x+size,y:y+size]
    return slices
def generate_slice1(name_x,name_y,data_path,size):
    data_dir=os.listdir(data_path)
    data_x=[]
    data_y=[]
    load_y=""
    for file in data_dir:
        if "MatK" in file:
            f = file.replace("MatK","")
            index = f.split('.')[0]
            load_y=str(int(index)) +" part_per_bin.dat"
            tmp=load_dataset_splice(data_path+"/"+file,data_path+"/"+load_y,0)
            data_x.append(get_a_subarray(tmp[0],size))
            data_y.append(get_a_subarray(tmp[1],size))
            print(len(data_x))
                    
    for x in range(len(data_y)):
        for y in range(len(data_y[x])):
            if 0 in data_y[x][y].shape or 0 in data_x[x][y].shape:
                data_x[x].pop(y)
                data_y[x].pop(y)
        
    return(data_x,data_y)
def generate_dataset_1slice(name_x,name_y,data_path,save_dir,step_size=0):
    tmp=generate_slice1(name_x,name_y,data_path,step_size)
    tmp = normlize_data(tmp)
    x_train,x_test,y_train,y_test=train_test_split(tmp[0],tmp[1],test_size=0.1,random_state=42)
    for x in range(len(x_train)):
        np.save(save_dir+"\\train\\MatK"+str(x)+".npy",x_train[x])
    for y in range(len(y_train)):
        np.save(save_dir+"\\train\\"+str(y)+" part_per_bin"+".npy",y_train[y])
    for x in range(len(x_test)):
        np.save(save_dir+"\\test\\MatK"+str(x)+".npy",x_test[x])
    for y in range(len(y_test)):
        np.save(save_dir+"\\test\\"+str(y)+" part_per_bin"+".npy",y_test[y])
def convert_to_npy(name_x,name_y,data_path,save_dir):
    data_dir=os.listdir(data_path)
    data=[[],[]]
    load_y=""
    for file in data_dir:
        if "MatK" in file:
            index = ""
            for c in file:
                if c.isnumeric():
                   index+=c
            tmp=""  
            add=False              
            for x in range(len(index)):
                if index[x] !="0":
                    add=True
                if add:
                    tmp+=index[x]
            index=tmp
            for file_y in data_dir:
                if ("part_per_bin" in file_y) and (index in file_y):
                    load_y=file_y
            tmp=load_dataset_splice(data_path+"/"+file,data_path+"/"+load_y,0)
            tmp = normlize_data(tmp)
            data[0]+=[tmp[0]]
            data[1]+=[tmp[1]]
    print("\n"+str(tmp[0]))
    #print(len(data[0]))
    #print(len(data[1]))
    x_train,x_test,y_train,y_test=train_test_split(data[0],data[1],test_size=0.1,random_state=42 )
    for x in range(len(x_train)):
        np.save(save_dir+"\\train\\MatK"+str(x)+".npy",x_train[x])
    for y in range(len(y_train)):
        np.save(save_dir+"\\train\\"+str(y)+" part_per_bin"+".npy",y_train[y])
    for x in range(len(x_test)):
        np.save(save_dir+"\\test\\MatK"+str(x)+".npy",x_test[x])
    for y in range(len(y_test)):
        np.save(save_dir+"\\test\\"+str(y)+" part_per_bin"+".npy",y_test[y])
def load_data_sliced(name_x,name_y,data_path,step_size=0):
    data_dir=os.listdir(data_path)
    data_x=[]
    data_y=[]
    load_y=""
    for file in data_dir:
        if "MatK" in file:
            f = file.replace("MatK","")
            index = f.split('.')[0]
            load_y=str(int(index)) +" part_per_bin.dat"
            tmp=load_dataset_splice(data_path+"/"+file,data_path+"/"+load_y,0)
            data_x.append(get_subarrays(tmp[0],step_size))
            data_y.append(get_subarrays(tmp[1],step_size))
                    
    for x in range(len(data_y)):
        for y in range(len(data_y[x])):
            if 0 in data_y[x][y].shape or 0 in data_x[x][y].shape:
                data_x[x].pop(y)
                data_y[x].pop(y)
        
    return(data_x,data_y)
def get_subarrays(array,size):
    x_end_index=array.shape[0]
    y_end_index=array.shape[1]-size
    slices=[]
    for y in range(0,y_end_index,int(size/2)):
        for x in range(0,x_end_index,int(size/2)):
            slices+=[array[x:x_end_index,y:y+size]]
    return slices
def normlize_data(tmp):
    biggest=0
    for y in tmp[1]:
        for x in y:
            if x.max() > biggest:
                biggest=x.max()
    for y in range(len(tmp[1])):
        for x in range(len(tmp[1][y])):
            tmp[1][y][x]=tmp[1][y][x]/biggest
    return tmp
def generate_dataset(name_x,name_y,data_path,save_dir,step_size=0):
    tmp=load_data_sliced(name_x,name_y,data_path,step_size)
    tmp = normlize_data(tmp)
    x_train,x_test,y_train,y_test=train_test_split(tmp[0],tmp[1],test_size=0.1,random_state=42)
    for x in range(len(x_train)):
        for i in range(len(x_train[x])):
            np.save(save_dir+"\\train\\MatK"+str(x)+","+str(i)+".npy",x_train[x][i])
    for y in range(len(y_train)):
        for i in range(len(y_train[y])):
            np.save(save_dir+"\\train\\"+str(y)+","+str(i)+" part_per_bin"+".npy",y_train[y][i])
    for x in range(len(x_test)):
        for i in range(len(x_test[x])):
            np.save(save_dir+"\\test\\MatK"+str(x)+","+str(i)+".npy",x_test[x][i])
    for y in range(len(y_test)):
        for i in range(len(y_test[y])):
            np.save(save_dir+"\\test\\"+str(y)+","+str(i)+" part_per_bin"+".npy",y_test[y][i])
def load_dataset_splice(file_x,file_y,rand_mag):
    tmp=_load_internal(file_x, file_y)
    
    #define variables
    #to_tensor = transforms.Compose([transforms.ToTensor()])
    data_x=[]
    data_y=[]
    #load fiels and store with label-make load multiple
    for x in tmp[0]:
        for y in x:
            y+=random.uniform(-rand_mag, rand_mag)
    if not(0 in tmp[0].shape):
        data_x=tmp[0]
        data_y=tmp[1]
    return(data_x,data_y)
def load_data_np(name_x,name_y,data_path):
    data_dir=os.listdir(data_path)
    data_x=[]
    data_y=[]
    load_y=""
    for file in data_dir:
        if "MatK" in file:
            f = file.replace("MatK","")
            index = f.split('.')[0]
            
            '''
            index = ""
            for c in file:
                if c.isnumeric():
                   index+=c
            tmp=""  
            add=False              
            for x in range(len(index)):
                if index[x] !="0":
                    add=True
                if add:
                    tmp+=index[x]
            index=tmp
            '''
            load_y=index +" part_per_bin.npy"
            tmp=load_dataset_full_np(data_path+"/"+file,data_path+"/"+load_y,0)
            data_x.append(tmp[0])
            data_y.append(tmp[1])
  
    for x in range(len(data_y)):
        if 0 in data_y[x].shape or 0 in data_x[x].shape:
            data_x.pop(x)
            data_y.pop(x)
    return(data_x,data_y)
def load_dataset_full_np(file_x,file_y,rand_mag):
    tmp=_load_internal_np(file_x, file_y)
    #define variables
    #to_tensor = transforms.Compose([transforms.ToTensor()])
    #load fiels and store with label-make load multiple
    #for x in tmp[0]:
        #for y in x:
            #y+=random.uniform(-rand_mag, rand_mag)
    data_x=torch.from_numpy(tmp[0]).type(torch.float32)
    data_y=tmp[1]
    return(data_x,data_y)
def _load_internal_np(file_x,file_y): #internal function for reading row data file
    #load file
    t=np.load(file_x) 
    #setup progress bar
    bar_widgets=[progressbar.FormatLabel("loading data")," ",progressbar.Percentage(), " ", progressbar.GranularBar(markers=" ▁▂▃▄▅▆▇█")," ",progressbar.AdaptiveETA(),]
    bar_max=t.size
    bar = progressbar.ProgressBar(max_value=bar_max,widgets=bar_widgets)
    counter=0
    #load expected output file
    Y=np.load(file_y)
    #return loaded data
    return[t,Y]
def load_data_mixed(name_x,name_y,data_path):
    data_dir=os.listdir(data_path)
    data_x=[]
    data_y=[]
    load_y=""
    for file in data_dir:
        print(file)
        if "MatK" in file:

            index = ""
            for c in file:
                if c.isnumeric():
                   index+=c
            tmp=""  
            add=False              
            for x in range(len(index)):
                if index[x] !="0":
                    add=True
                if add:
                    tmp+=index[x]
            index=tmp
            for file_y in data_dir:
                if ("part_per_bin" in file_y) and (index in file_y):
                    load_y=file_y
            print(str(load_y))
            tmp=load_dataset_np(data_path+"/"+file,data_path+"/"+load_y,0)
            data_x+=tmp[0]
            data_y+=tmp[1]
    for x in range(len(data_y)):
        if 0 in data_y[x].shape or 0 in data_x[x].shape:
            data_x.pop(x)
            data_y.pop(x)
    return(data_x,data_y)
def load_dataset_np(file_x,file_y,rand_mag):
    tmp=_load_internal(file_x, file_y)
    #define variables
    #to_tensor = transforms.Compose([transforms.ToTensor()])
    data_x=[]
    data_y=[]
    #load fiels and store with label-make load multiple
    for x in tmp[0]:
        for y in x:
            y+=random.uniform(-rand_mag, rand_mag)
    data_x+=[torch.from_numpy(tmp[0]).reshape([1,*tmp[0].shape]).type(torch.float32)]
    data_y+=[tmp[1]]

    return(data_x,data_y)
def load_full(x_dir,y_dir,rand_mag):
    files_x=os.listdir(x_dir)
    files_y=os.listdir(y_dir)
    data_x=[]
    data_y=[]
    for file in files_x:
        tmp=load_dataset_np(x_dir+"/"+file,y_dir+"/part_per_bin"+file.replace(".txt","")[-1]+".dat",rand_mag)
        data_x=tmp[0]
        data_y=tmp[1]
    return(data_x,data_y)
#generate a spesific number of data samples 
def generate_samples(file_x,file_y,sample_num,size_min,size_max,loading_bar=False,noise_strength=0,identifier=""):
    #load data files
    tmp_lod=_load_internal(file_x,file_y)
    X=tmp_lod[0]
    Y=tmp_lod[1]
    #setup progress bar
    if loading_bar:
        bar_widgets=[progressbar.FormatLabel("generating samples")," ",progressbar.Percentage(), " ", progressbar.GranularBar(markers=" ▁▂▃▄▅▆▇█")," ",progressbar.AdaptiveETA(),]
        bar_max=sample_num
        bar = progressbar.ProgressBar(max_value=bar_max,widgets=bar_widgets)
        for x in range(sample_num):
            #generate pre detemiend number of samples
            bar.update(x)
            _generate_sample(str(x)+identifier,size_min,size_max,X,Y,noise_strength)
    else:
        for x in range(sample_num):
            #generate pre detemiend number of samples
            _generate_sample(x,size_min,size_max,X,Y)
def delete_dataset():
    y_n=input("Are you sure you want to delete the dataset?[y/n]")
    if y_n.lower() == "y" or y_n.lower() == "yes":
        if os.path.isdir('dataset'):
            shutil.rmtree('dataset')
def _load_internal(file_x,file_y): #internal function for reading row data file
    #limit reading to 300 rows to avoid error
    reading_list=[]
    for x in range(300):
        reading_list+=[x]
    #load file
    t=np.loadtxt(file_x,dtype=str) 
    #setup progress bar
    bar_widgets=[progressbar.FormatLabel("loading data")," ",progressbar.Percentage(), " ", progressbar.GranularBar(markers=" ▁▂▃▄▅▆▇█")," ",progressbar.AdaptiveETA(),]
    bar_max=t.size
    bar = progressbar.ProgressBar(max_value=bar_max,widgets=bar_widgets)
    counter=0
    #cconvert to valid floats from sientific notation
    for x in np.nditer(t):
        counter+=1
        bar.update(counter)
        tmp_x=np.array2string(x).replace("\'","")
        x=_fix_num(tmp_x)
    t=t.astype(float)
    for n in t:
        for m in n:
            if m != m:
                m=0
    #load expected output file
    Y=np.loadtxt(file_y,dtype=float)
    #return loaded data
    return[t,Y]
def _fix_num(num_str): #transform from sientific notation to valid float
    num_str=num_str.replace("e","*math.exp(")
    num_str+=")"
    num_str=_fix_leading_0(num_str)
    return eval(num_str)
def _fix_leading_0(num_str):
    for x in range(len(num_str)):#remove leading zero
        if num_str[x]=="0" and num_str[x-1].isnumeric()==False and num_str[x-1] != ".":
            return num_str[:x]+num_str[x+1:]
    return num_str
def _generate_sample(data_identifier,size_min,size_max,X,Y,noise_strength): #generate a single sample from given data
    sign_arr=[1,-1]
    #choose random section of data
    sample_size=5#sample_size=np.random.randint(size_min,size_max)
    sample_pos_x=np.random.randint(0,X.shape[0]-sample_size)
    sample_pos_y=np.random.randint(0,X.shape[1]-sample_size)
    #copy selected section into array
    #sample_arr=np.empty(X[(sample_pos_x-sample_size):sample_pos_x,(sample_pos_y-sample_size):sample_pos_y])
    #out_arr=np.empty(Y[(sample_pos_x-sample_size):sample_pos_x,(sample_pos_y-sample_size):sample_pos_y])
    sample_arr=np.empty((sample_size,sample_size))
    out_arr=np.empty((sample_size,sample_size))
    for x in range(sample_size):
        for y in range(sample_size):
            x_cord=sample_pos_x+x
            y_cord=sample_pos_y+y
            #print(str(sample_pos_x)+","+str(sample_pos_y)) -debuging
            sample_arr[x,y]=X[x_cord,y_cord]
            out_arr[x,y]=Y[x_cord,y_cord]
            #print(sample_arr[x,y]) -debuging
            if noise_strength!=0:
                np.random.shuffle(sign_arr)
                sample_arr[x,y]+=(np.random.randint(0,noise_strength)/10)*(sign_arr[0])
                #print(str(sample_arr[x,y])+"--") - debuging
    #encode input array as image
    s_img = MinMaxScaler().fit_transform(sample_arr)
    s_img=s_img*255
    s_img=np.uint8(s_img)
    in_img = im.fromarray(s_img)
    in_img=in_img.convert("L")
    #fetch expected output data 
    #save generated sample and expected output
    if os.path.isdir('dataset')==False:
        os.mkdir("dataset")
        os.mkdir("dataset/X_data")
        os.mkdir("dataset/y_data")
    if os.path.isdir('dataset/X_data')==False:
        os.mkdir("dataset/X_data")
    if os.path.isdir('dataset/y_data')==False:
        os.mkdir("dataset/y_data")
    in_img.save("dataset/X_data/sample_in_"+str(data_identifier)+"("+str(sample_size)+","+str(sample_size)+")"+".png")
    np.save("dataset/y_data/sample_out_"+str(data_identifier)+"("+str(sample_size)+","+str(sample_size)+")",out_arr)

#testing
#generate_samples("files/input/MatK0001.txt","Files\Output\part_per_bin1.dat",1,3,10,True,0)
#dataset=load_dataset("dataset/X_data","dataset/y_data")#"files/input/MatK0001.txt","Files\Output\part_per_bin1.dat"
#print(dataset[1]["sample_0(4,4)"])