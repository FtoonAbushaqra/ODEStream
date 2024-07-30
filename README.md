# ODEStream: A Buffer-Free Online Learning Framework with ODE-based Adaptor for Streaming Time Series Forecasting

**Note:** We are still working on organizing this code. Some features and documentation may be incomplete or subject to change.


This repository contains the code for our state-of-the-art ODE-based model for countinual learning. The code is written in PyTorch and follows the methodology described in our paper


## Model Overview

![Model Architecture](Framework.png)



## Datasets
All datasets used are in the Dataset folder. 

The learn representation generated using ODE-RNN is saved in `DataRep/` fplder. 
To regenerate the we recommend you follwoing the instructions of the original code repository

Tha attention and payramid module codes are in scr. 


to generate the final prediction 
Run SeqLink.py


## Repository Structure

- `src/`: Contains the source code for the model, attention mechanism, and pyramid sorting.
- `data/`: Example data files used for training and testing.
- `notebooks/`: Jupyter notebooks demonstrate the use of the model and visualize results.
- `docs/`: Documentation and additional resources.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt


