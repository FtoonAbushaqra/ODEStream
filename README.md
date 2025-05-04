# ODEStream: A Buffer-Free Online Learning Framework with ODE-based Adaptor for Streaming Time Series Forecasting

**Note:**  Some features and documentation may be subject to change.


This repository contains the code for our state-of-the-art ODE-based model for continual learning. The code is written in PyTorch and follows the methodology described in our paper.

**Paper**: [ODEStream]([https://link.to/your-paper](https://arxiv.org/abs/2411.07413))

**Reviewed on OpenReview**: [https://openreview.net/forum?id=TWOTKhwU5n](https://openreview.net/forum?id=TWOTKhwU5n)


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
@article{DBLP:journals/corr/abs-2411-07413,
  author       = {Futoon M. Abushaqra and
                  Hao Xue and
                  Yongli Ren and
                  Flora D. Salim},
  title        = {ODEStream: {A} Buffer-Free Online Learning Framework with ODE-based
                  Adaptor for Streaming Time Series Forecasting},
  journal      = {CoRR},
  volume       = {abs/2411.07413},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2411.07413},
  doi          = {10.48550/ARXIV.2411.07413},
  eprinttype    = {arXiv},
  eprint       = {2411.07413},
  timestamp    = {Wed, 01 Jan 2025 11:02:42 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2411-07413.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```




