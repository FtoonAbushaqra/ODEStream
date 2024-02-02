import math
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as npr
sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader , TensorDataset
import torch
from torch.utils.data import Dataset
import numpy as np

# Define a synthetic dataset class

from torch.utils.data import Dataset

from numpy import array

def split_sequence2(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X)  , array(y)

def cutout(data, percentage=0.3):

    if not (0 < percentage <= 1):
        raise ValueError("Percentage should be between 0 and 1.")

    num_rows, num_columns = data.shape
    total_values = num_rows * (num_columns - 1)  # Exclude the last column

    num_values_to_remove = int(percentage * total_values)

    flattened_indices = np.random.choice(total_values, size=num_values_to_remove, replace=False)
    indices_x = flattened_indices // (num_columns - 1)
    indices_j = flattened_indices % (num_columns - 1)

    # Increment the column index if it's greater than or equal to the excluded column
    indices_j[indices_j >= (num_columns - 1)] += 1

    data[indices_x, indices_j] = 0

    return data

def get_time(lag, x,val_x=None ):
    n_steps=lag
    timem = np.linspace(0, n_steps, num=n_steps)
    timem = np.hstack([timem[:, None]] * x.shape[0])  # 358 samples
    timem = torch.from_numpy(timem[:, :, None]).to(torch.float32)
    timem = timem.transpose(0, 1)
    samp_ts = timem
    # print("samp_ts " , samp_ts.shape)
    if val_x ==None:
        return samp_ts
    else:
        timem = np.linspace(0, n_steps, num=n_steps)
        timem = np.hstack([timem[:, None]] * val_x.shape[0])  # 358 samples
        timem = torch.from_numpy(timem[:, :, None]).to(torch.float32)
        timem = timem.transpose(0, 1)
        val_samp_ts = timem
        return samp_ts, val_samp_ts

def get_data(dataset, flag, lag, task, reguler):
     path = "data/"
     data = read_csv(path+dataset+".csv", header=0,
                   parse_dates=[0], index_col=0, squeeze=True)
     data.fillna(data.mean(), inplace=True)
     data[data >= 1E308] = 0
     data = data.values

     if dataset == 'ETTm1':
         train_size = 2880
         val_from = train_size - lag #2856
         val_to = 3600
         test_from= val_to - lag
     if dataset in ['ETTh1' , 'ETTh2']:
         train_size = 2880
         val_from = train_size - lag #2856
         val_to = 3600
         test_from= val_to - lag

     if dataset == 'ECL':
         train_size = 5260
         val_from = train_size - lag
         val_to = 6570
         test_from = 6552#val_to - lag
         #testf = 19752
     if dataset == 'WTH':
         train_size = 7012
         val_from = train_size - lag
         val_to = 8766
         test_from = val_to - lag

     #scaler = MinMaxScaler(feature_range=(0, 1))
     scaler = StandardScaler()
     if flag=="initial":
        if reguler=='f':
            data = cutout(data)

        if task =='m' or task =='ms':
            x = data [:,: ] # client ID
            traind = data[ :train_size,:] #trian test with val set  with client id
            scaler.fit(traind)
            xd = scaler.transform(x)
        if task =='s':
            x = data[:, -1]  # client ID
            traind = data[:train_size, -1]  # trian test with val set  with client id
            traind = traind.reshape(-1, 1)
            scaler.fit(traind)
            x = x.reshape(-1, 1)
            xd = scaler.transform(x)


        x = xd[ :train_size,:]
        val_x = xd[ val_from:val_to,:]
        x, y = split_sequence2(x, lag)
        val_x, val_y = split_sequence2(val_x, lag)
        if task == "ms":
            y=y[:,-1:]
            val_y = val_y[:,-1:]

        x = torch.from_numpy(x[:, :, ]).to(torch.float32)
        y = torch.from_numpy(y[:, :, ]).to(torch.float32)
        val_x = torch.from_numpy(val_x[:, :, ]).to(torch.float32)
        val_y = torch.from_numpy(val_y[:, :, ]).to(torch.float32)
        samp_ts, val_samp_ts= get_time(lag, x, val_x)
        return x, y , val_x, val_y, samp_ts, val_samp_ts

     if flag == "stream" :
         if reguler == 'f':
             data = cutout(data)
         if task == 'm' or task == 'ms':
            x = data[:, :]
            traind = data[:train_size, :]
            scaler.fit(traind)
            x = scaler.transform(x)
         if task == 's' :
             x = data[:, -1]
             traind = data[:train_size, -1]
             traind = traind.reshape(-1, 1)
             scaler.fit(traind)
             x = x.reshape(-1, 1)
             x = scaler.transform(x)

         x = x[test_from:, :]
         x, y = split_sequence2(x, lag)
         x = torch.from_numpy(x[:, :, ]).to(torch.float32)
         y = torch.from_numpy(y[:, :, ]).to(torch.float32)
         if task == "ms":
             y = y[:, -1:]

         samp_ts  = get_time(lag, x)
         return x, y,  samp_ts,



def gen_batch(batch_size,x,y, samp_ts, lag ):
    n_sample = lag-1
    # n_batches = samp_trajs.shape[1] // batch_size
    # time_len = samp_trajs.shape[0]
    x = x.transpose(0, 1)
    samp_ts = samp_ts.transpose(0, 1)
    n_batches = x.shape[1] // batch_size
    time_len = x.shape[0]
    #print ("beginnign : " , x.shape)
    n_sample = min(n_sample, time_len)
    #print("n_batches", n_batches)
    #print("time_len", time_len)
    #print("n_sample", n_sample)
    #print("batch_size", batch_size)
    for i in range(n_batches):
        if n_sample > 0:
            #print ("n_sample > 0")
            t0_idx = npr.multinomial(1, [1. / (time_len - n_sample)] * (time_len - n_sample))
            t0_idx = np.argmax(t0_idx)
            tM_idx = t0_idx + n_sample+1
            #print (t0_idx , t0_idx , tM_idx)
        else:
            print("n_sample <>> 0")
            t0_idx = 0
            tM_idx = time_len
        #print("beginnign : ", x.shape)
        frm, to = batch_size * i, batch_size * (i + 1)
        #print (frm, to)
        # yield samp_trajs[t0_idx:tM_idx, frm:to], samp_ts[t0_idx:tM_idx, frm:to]
        #print ("samp_ts in getn bahc ", samp_ts[t0_idx:tM_idx, frm:to].shape)
        #print("x in getn bahc ", x.shape)
        #print("x in getn bahc ", x[t0_idx:tM_idx, frm:to].shape)

        yield x[t0_idx:tM_idx, frm:to], samp_ts[t0_idx:tM_idx, frm:to], y[frm:to]

#print ("x: ",x.shape)


