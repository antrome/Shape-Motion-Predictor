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

class Conv1D(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        :param num_securities: int, number of stocks
        :param hidden_size: int, size of hidden layers
        :param dilation: int, dilation value
        :param T: int, number of look back points
        """
        super(Conv1D, self).__init__()
        self.in_channels = input_channel
        self.out_channels = output_channel
        # First Layer
        self.conv1 = nn.Conv1d(input_channel, output_channel, kernel_size=1)
        self.relu = nn.ReLU()

        # Output layer
        self.conv_final = nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)

    def forward(self, x):
        """
        :param x: Pytorch Variable, batch_size x n_stocks x T
        :return:
        """

        x = x.permute(0,2,1)

        # First layer
        out = self.conv1(x)
        out = self.relu(out)

        # Final layer
        out = self.conv_final(out)
        out = out[:, :, :]

        out = out+x

        out = out.permute(0,2,1)

        return out


