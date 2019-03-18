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
    def __init__(self, input_dim, hidden_dim, layer_dim, dropout):
        super(ResLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.encoder = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True,dropout=dropout)

        self.decoder = nn.LSTM(self.hidden_dim, self.input_dim, self.layer_dim, batch_first=True,dropout=dropout)

    def forward(self, input):
        # Encode
        _, (last_hidden, _) = self.encoder(input)
        # It is way more general that way
        encoded = last_hidden.repeat(input.shape)

        # Decode
        y, _ = self.decoder(encoded)
        return torch.squeeze(y)