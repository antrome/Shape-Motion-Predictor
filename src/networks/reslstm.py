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

class ResLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, input_rows, input_cols,dropout,seq_dim):
        super(ResLSTM, self).__init__()
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

        # Dropout
        self.dropout = dropout

        # Sequence dimension
        self._seq_dim = seq_dim

        # Building your LSTM
        self.encoder = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True,dropout=dropout)
        self.decoder = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True,dropout=dropout)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        xh = x[:,:50,:]

        x = x.reshape(x.size(0),x.size(1),self.input_rows,self.input_cols)
        xh = xh.reshape(xh.size(0),xh.size(1),self.input_rows,self.input_cols)

        # reshape
        out = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))
        outh = xh.reshape(xh.size(0), xh.size(1) * xh.size(2), xh.size(3))

        out = out.reshape(x.size(0), x.size(1), x.size(2)*x.size(3))
        outh = outh.reshape(xh.size(0), xh.size(1), xh.size(2)*xh.size(3))

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        h0h = torch.zeros(self.layer_dim, xh.size(0), self.hidden_dim).requires_grad_().to(xh.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0h = torch.zeros(self.layer_dim, xh.size(0), self.hidden_dim).requires_grad_().to(xh.device)

        #Flatten Parameters
        self.encoder.flatten_parameters()

        # One time step
        outh, (hnh, cnh) = self.encoder(outh, (h0h, c0h))
        #out, (hn, cn) = self.lstm(out, (h0, c0))

        out, (hn, cn) = self.decoder(out, (hnh, cnh))

        outFrames = []
        for f in range(out.size(1)):
            outFrames.append(self.fc(out[:, f, :]))
            # Index hidden state of last time step

        out = torch.stack(outFrames,dim=1)

        out = out.reshape(x.size(0), x.size(1), x.size(2), x.size(3))

        out = out + x

        out = out.reshape(x.size(0),x.size(1),self.output_dim)

        return out