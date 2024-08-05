# ODEStream: A Buffer-Free Online Learning Framework with ODE-based Adaptor for Streaming Time Series Forecasting

**Note:** We are still working on organizing this code. Some features and documentation may be incomplete or subject to change.


This repository contains the code for our state-of-the-art ODE-based model for continual learning. The code is written in PyTorch and follows the methodology described in our paper


## Model Overview

![Model Architecture](Framework.png)



## Datasets
All datasets used are in the Dataset folder. 

## Requirements
- pytorch == 1.12


## Arguments
- lag --> lockback window size
- datasetname: 'ECL, 'ETTm1', 'ETTh1', 'ETTh2', 'WTH' --> dataste name 
- task: 's', 'ms' , 'm'  
- flag:  'initial', 'stream'  --> initial training ( model warm-up), or online learning 
- savedmodelpath = "models/"
- resultpath = "results/"
- mempath = "memory/"
- reguler = 't'







