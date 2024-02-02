
import math
import seaborn as sns
sns.color_palette("bright")
from torch import Tensor
import torch.nn as nn
import torch
import numpy as np
from memory_profiler import profile

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        out, _ = self.lstm(x, (h0, c0))

        # Index the last time step output
        out = out[-1:,: , :]
        #print(out.shape)
        out = out.reshape(out.size(1), out.size(2))
        # Pass the output through the fully connected layer
        out = self.fc(out)

        return out

class ConcatenationLayer(nn.Module):
    def __init__(self):
        super(ConcatenationLayer, self).__init__()

    def forward(self, x1, x2):
        # Concatenate the outputs of both models along the specified dimension (dim=1)
        concatenated_output = torch.cat((x1, x2), dim=1)
        return concatenated_output

class CombinedModel(nn.Module):
    def __init__(self, model1, model2, concatenation_layer, size, output_size):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.concatenation_layer = concatenation_layer
        self.final_layer = nn.Linear(output_size*2 , output_size)

    @profile
    def forward(self, x,t,yr):
        output1 , z, z_mean, z_log_var= self.model1(x,t,yr)

       # print(z.shape)
       # print(z_mean.shape)
       # print(z_log_var.shape)
      #  exit()
        output2 = self.model2(x)

        concatenated_output = self.concatenation_layer(output1, output2)


        concatenated_output = self.final_layer(concatenated_output)
        return concatenated_output,output1 , output2,  z, z_mean, z_log_var