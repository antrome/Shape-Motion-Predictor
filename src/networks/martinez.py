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
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import math
import os

class MartinezSimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,dropout,seq_dim,pred_dim):
        super(MartinezSimpleLSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Output dimensions
        self.output_dim = output_dim

        # Dropout
        self.dropout = dropout

        # Sequence dimension
        self._seq_dim = seq_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True,dropout=dropout)
        #Flatten Parameters
        self.lstm.flatten_parameters()

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # One time step
        out, (hn, cn) = self.lstm(x, (h0, c0))
        outFrames = []

        for f in range(out.size(1)):
            outFrames.append(self.fc(out[:, f, :]))
            # Index hidden state of last time step

        out = torch.stack(outFrames,dim=1)

        out = out + x

        return out

class MartinezSimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,dropout,seq_dim,pred_dim):
        super(MartinezSimpleGRU, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Output dimensions
        self.output_dim = output_dim

        # Dropout
        self.dropout = dropout

        # Sequence dimension
        self._seq_dim = seq_dim

        self.lstm = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True,dropout=dropout)
        #Flatten Parameters
        self.lstm.flatten_parameters()

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # One time step
        out, hn = self.lstm(x, h0)
        outFrames = []

        for f in range(out.size(1)):
            outFrames.append(self.fc(out[:, f, :]))
            # Index hidden state of last time step

        out = torch.stack(outFrames,dim=1)

        out = out + x

        return out

class MartinezSimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,dropout,seq_dim,pred_dim):
        super(MartinezSimpleRNN, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Output dimensions
        self.output_dim = output_dim

        # Dropout
        self.dropout = dropout

        # Sequence dimension
        self._seq_dim = seq_dim

        self.lstm = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,dropout=dropout)
        #Flatten Parameters
        self.lstm.flatten_parameters()

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # One time step
        out, hn = self.lstm(x, h0)
        outFrames = []

        for f in range(out.size(1)):
            outFrames.append(self.fc(out[:, f, :]))
            # Index hidden state of last time step

        out = torch.stack(outFrames,dim=1)

        out = out + x

        return out

class MartinezLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,dropout,seq_dim,pred_dim):
        super(MartinezLSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Dropout
        self.dropout = dropout

        # Sequence dimension
        self.seq_dim = seq_dim

        # Frame from which the prediction is made
        self.pred_dim = pred_dim

        self.lstm_encoder = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)
        self.lstm_encoder.flatten_parameters()

        self.fc_in = nn.Linear(input_dim, hidden_dim)

        self.lstm_decoder_cells = nn.ModuleList()
        for l in range(layer_dim):
            self.lstm_decoder_cells.append(nn.LSTMCell(hidden_dim, hidden_dim))

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # One time step
        _, (h_enc, c_enc) = self.lstm_encoder(x[:, :self.pred_dim, :], (h0, c0))

        outFrames = []
        hs = [h_enc[l, ...] for l in range(self.layer_dim)]
        cs = [c_enc[l, ...] for l in range(self.layer_dim)]
        outFrame = x[:, self.pred_dim -1, :]
        for f in range(self.seq_dim - self.pred_dim):

            outFrame_n = self.fc_in(outFrame)

            for l in range(self.layer_dim):
                hs[l], cs[l] = self.lstm_decoder_cells[l](outFrame_n, (hs[l], cs[l]))
                outFrame_n = hs[l]

            outFrame = self.fc_out(outFrame_n) + outFrame
            outFrames.append(outFrame)
            # Index hidden state of last time step

        out = torch.stack(outFrames,dim=1)

        out = torch.cat([x[:, :self.pred_dim, :],out],dim=1)

        return out

class MartinezGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,dropout,seq_dim,pred_dim):
        super(MartinezGRU, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Dropout
        self.dropout = dropout

        # Sequence dimension
        self.seq_dim = seq_dim

        # Frame from which the prediction is made
        self.pred_dim = pred_dim

        self.lstm_encoder = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)
        self.lstm_encoder.flatten_parameters()

        self.fc_in = nn.Linear(input_dim, hidden_dim)

        self.lstm_decoder_cells = nn.ModuleList()
        for l in range(layer_dim):
            self.lstm_decoder_cells.append(nn.GRUCell(hidden_dim, hidden_dim))

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # One time step
        _, h_enc = self.lstm_encoder(x[:, :self.pred_dim, :], h0)

        outFrames = []
        hs = [h_enc[l, ...] for l in range(self.layer_dim)]
        outFrame = x[:, self.pred_dim -1, :]
        for f in range(self.seq_dim - self.pred_dim):

            outFrame_n = self.fc_in(outFrame)

            for l in range(self.layer_dim):
                hs[l] = self.lstm_decoder_cells[l](outFrame_n, (hs[l]))
                outFrame_n = hs[l]

            outFrame = self.fc_out(outFrame_n) + outFrame
            outFrames.append(outFrame)
            # Index hidden state of last time step

        out = torch.stack(outFrames,dim=1)

        out = torch.cat([x[:, :self.pred_dim, :],out],dim=1)

        return out

class ERDGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,dropout,seq_dim,pred_dim,res_conn):
        super(ERDGRU, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Output dimensions
        self.output_dim = output_dim

        # Dropout
        self.dropout = dropout

        # Sequence dimension
        self._seq_dim = seq_dim

        self.res_conn = res_conn

        # Encoder
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(2e-1, True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.lstm = nn.GRU(hidden_dim, hidden_dim, layer_dim, batch_first=True,dropout=dropout)
        #Flatten Parameters
        self.lstm.flatten_parameters()

        # Decoder
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(2e-1, True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):

        def _apply_module(module, seq):
            outFrames = []
            for f in range(seq.size(1)):
                outFrames.append(module(seq[:, f, :]))
            seq = torch.stack(outFrames, dim=1)
            return seq

        out = _apply_module(self.fc_in, x)

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, hn = self.lstm(out, h0)

        out = _apply_module(self.fc_out, out)

        if self.res_conn:
            out = out + x

        return out

class ERDLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,dropout,seq_dim,pred_dim,res_conn):
        super(ERDLSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Output dimensions
        self.output_dim = output_dim

        # Dropout
        self.dropout = dropout

        # Sequence dimension
        self._seq_dim = seq_dim

        self.res_conn = res_conn

        # Encoder
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(2e-1, True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.lstm = nn.GRU(hidden_dim, hidden_dim, layer_dim, batch_first=True,dropout=dropout)
        #Flatten Parameters
        self.lstm.flatten_parameters()

        # Decoder
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(2e-1, True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):

        def _apply_module(module, seq):
            outFrames = []
            for f in range(seq.size(1)):
                outFrames.append(module(seq[:, f, :]))
            seq = torch.stack(outFrames, dim=1)
            return seq

        out = _apply_module(self.fc_in, x)

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, (hn,cn) = self.lstm(out, (h0,c0))

        out = _apply_module(self.fc_out, out)

        if self.res_conn:
            out = out + x

        return out

