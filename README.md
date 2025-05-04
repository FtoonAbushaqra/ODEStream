# ODEStream: A Buffer-Free Online Learning Framework with ODE-based Adaptor for Streaming Time Series Forecasting

**Note:**  Some features and documentation may be subject to change.


This repository contains the code for our state-of-the-art ODE-based model for continual learning. The code is written in PyTorch and follows the methodology described in our paper

## Abstract

Addressing the challenges of irregularity and concept drift in streaming time series is crucial for real-world predictive modelling. Previous studies in time series continual learning often propose models that require buffering long sequences, potentially restricting the responsiveness of the inference system. Moreover, these models are typically designed for regularly sampled data, an unrealistic assumption in real-world scenarios. This paper introduces ODEStream, a novel buffer-free continual learning framework that incorporates a temporal isolation layer to capture temporal dependencies within the data. Simultaneously, it leverages the capability of neural ordinary differential equations to process irregular sequences and generate a continuous data representation, enabling seamless adaptation to changing dynamics in a data streaming scenario. Our approach focuses on learning how the dynamics and distribution of historical data change over time, facilitating direct processing of streaming sequences. Evaluations on benchmark real-world datasets demonstrate that ODEStream outperforms the state-of-the-art online learning and streaming analysis baseline models, providing accurate predictions over extended periods while minimising performance degradation over time by learning how the sequence dynamics change.


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



## Citation

If you use this code, please cite our paper:

```bibtex
@article{DBLP:journals/tmlr/Abushaqra0RS24,
  author       = {Futoon M. Abushaqra and
                  Hao Xue and
                  Yongli Ren and
                  Flora D. Salim},
  title        = {SeqLink: {A} Robust Neural-ODE Architecture for Modelling Partially
                  Observed Time Series},
  journal      = {Trans. Mach. Learn. Res.},
  volume       = {2024},
  year         = {2024},
  url          = {https://openreview.net/forum?id=WCUT6leXKf},
  timestamp    = {Thu, 08 Aug 2024 15:22:39 +0200},
  biburl       = {https://dblp.org/rec/journals/tmlr/Abushaqra0RS24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```




