# ODEStream: A Buffer-Free Online Learning Framework with ODE-based Adaptor for Streaming Time Series Forecasting

**Note:**  Some features and documentation may be subject to change.


This repository contains the code for our state-of-the-art ODE-based model for continual learning. The code is written in PyTorch and follows the methodology described in our paper.

**Paper**: [https://arxiv.org/abs/2411.07413](https://arxiv.org/abs/2411.07413)

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

## Results

### Table: Cumulative MSE Results

*Cumulative MSE results across several datasets using ODEStream and baselines for different tasks.*  
**Notation**: • univariate→univariate, * multivariate→univariate, ^ multivariate→multivariate

| Method        | ECL•       | ECL^     | ETTh1\*    | ETTh1^     | ETTh2\*    | ETTh2^     | ETTm1\*    | ETTm1^     | WTH\*      | WTH^       |
| ------------- | ---------- | -------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| RNN           | 0.576      | 33.84    | 43.36      | 1.2653     | 37.35      | 6.3711     | 41.08      | 0.4700     | 0.1636     | 0.4616     |
| ER            | 2.8142     | 2.8359   | 1.9785     | 0.2349     | 6.7558     | 0.5044     | 3.0550     | 0.0820     | 0.3138     | 0.1788     |
| DER++         | 2.8107     | **2.81** | 1.9712     | 0.2400     | 6.7380     | 0.5042     | 3.0467     | **0.0808** | 0.3097     | **0.1717** |
| FSnetNaive    | 2.9943     | 3.0533   | 2.0010     | 0.2296     | 6.7749     | 0.5033     | 3.0595     | 0.1143     | 0.3843     | 0.2462     |
| FSnet         | 2.8048     | 3.6002   | 1.9342     | 0.2814     | 6.6810     | 0.4388     | 3.0467     | 0.0866     | 0.3096     | 0.1633     |
| **ODEStream** | **0.1173** | 4.095    | **0.0594** | **0.1050** | **0.1640** | **0.1879** | **0.0625** | 0.2178     | **0.0441** | 0.2220     |


### Table: Multi-horizon Prediction Results

*Mean squared error (MSE) results across different forecasting steps (1, 7, and 24) for various datasets using ODEStream.*

| Steps | ECL    | ETTh1  | ETTh2  | ETTm1  | WTH    |
|-------|--------|--------|--------|--------|--------|
| 1     | 0.1173 | 0.0594 | 0.1640 | 0.0625 | 0.0441 |
| 7     | 0.4030 | 0.1006 | 0.4009 | 0.5997 | 0.1089 |
| 24    | 1.0030 | 0.1909 | 0.3751 | 0.9186 | 0.1330 |

### Table: Ablation Study Results

*MSE comparison showing the effect of different Temporal Input Layers (TILs) across datasets.*

| Method      | ECL    | ETTh1  | ETTh2  | ETTm1  | WTH    |
|-------------|--------|--------|--------|--------|--------|
| W/O TIL₁    | 0.299  | 0.127  | 0.372  | 0.100  | 0.200  |
| W/O TIL₂    | 0.369  | 0.084  | 0.347  | 0.098  | 0.178  |
| W/ TIL₁     | 0.119  | **0.054** | 0.176  | 0.090  | 0.060  |
| ODEStream   | **0.117** | 0.059  | **0.164** | **0.062** | **0.044** |


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




