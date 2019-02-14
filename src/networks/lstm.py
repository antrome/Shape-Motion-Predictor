import h5py
import numpy as np
import re
from os import walk
import glob
import scipy.io
import os, os.path
import sys
import torch
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as utils
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import math
import os

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, input_rows, input_cols):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Number of input rows for coordinates
        self.input_rows = input_rows

        # Number of input columns for coordinates
        self.input_cols = input_cols

        # Output dimensions
        self.output_dim = output_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        #Batch Normalization
        #self.bn1 = nn.BatchNorm1d(3)

    def forward(self, x):

        x = x.reshape(x.size(0),x.size(1),self.input_rows,self.input_cols)

        # reshape
        out = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))
        out = out.permute(0, 2, 1)

        #out = self.bn1(out)

        out = out.permute(0, 1, 2)

        out = out.reshape(x.size(0), x.size(1), x.size(2)*x.size(3))

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        #Flatten Parameters
        #self.lstm.flatten_parameters()

        # One time step
        out, (hn, cn) = self.lstm(out, (h0, c0))
        outFrames = []

        for f in range(out.size(1)):
            outFrames.append(self.fc(out[:, f, :]))
            # Index hidden state of last time step


        out = torch.stack(outFrames,dim=1)

        out = out.reshape(x.size(0), x.size(1), x.size(2), x.size(3))

        out = out + x

        out = out.reshape(x.size(0),x.size(1),self.output_dim)

        return out