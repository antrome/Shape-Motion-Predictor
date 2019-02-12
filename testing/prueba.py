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


class H36M(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self, hdf5file,transform=None):
        self.hdf5 = hdf5file
        self.transform = transform

    def __getitem__(self,item):

        with h5py.File(self.hdf5, 'r') as f:

            x_data = torch.from_numpy(f['{:03d}'.format(item)][:])

            #Pick only one each 2 frames
            x_data = x_data[:,::2,:,:]

        #Pick a random camera
        cam=random.randint(0, 3)
        frames=random.randint(0, x_data.shape[1]-101)
        x_data_cam_frame = x_data[cam][frames:frames+99][:][:]

        # Normalize your data here
        if self.transform:
            x_data_cam_frame = self.transform(x_data_cam_frame)

        labels_data_cam_frame = x_data[cam][frames+1:frames+100][:][:]
        self.tensor_x = torch.stack([torch.Tensor(i) for i in x_data_cam_frame])
        self.tensor_y = torch.stack([torch.Tensor(i) for i in labels_data_cam_frame])

        return self.tensor_x, self.tensor_y

    def __len__(self):
        with h5py.File(self.hdf5, 'r') as f:
            lens = len(list(f.keys()))
        return lens

'''
STEP 0: PREDEFINE TRANSFORMS
'''
transform = transforms.Compose([
        transforms.Normalize([0],[1])])

'''
STEP 1: LOADING DATASET
'''
#train_dataset = H36M("/media/finuxs/HDD/h36train.hdf5", transform=None)
train_dataset = H36M("../datasets/h36m/h36train.hdf5", transform=None)
#test_dataset = H36M("/media/finuxs/HDD/h36test.hdf5", transform=None)
#test_dataset = H36M("./datasets/h36m/h36test.hdf5", transform=None)


'''
STEP 2: MAKING DATASET ITERABLE
'''

batch_size = 32
n_iters = 450000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)

"""
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
"""

'''
STEP 3: CREATE MODEL CLASS
'''

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        #Batch Normalization
        #self.bn1 = nn.BatchNorm1d(input_dim)

    def forward(self, x):

        outFrames = []

        for f in range(x.size(1)):
            outFrames.append(x[:, f, :])
            # Index hidden state of last time step

        out = torch.stack(outFrames,dim=1)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        #Flatten Parameters
        self.lstm.flatten_parameters()

        # One time step
        out, (hn, cn) = self.lstm(out, (h0, c0))
        outFrames = []

        for f in range(out.size(1)):
            outFrames.append(self.fc(out[:, f, :]))
            # Index hidden state of last time step


        out = torch.stack(outFrames,dim=1)
        out = out + x

        return out

"""
# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, :, :])
        return out
"""

'''
STEP 4: INSTANTIATE MODEL CLASS
'''

input_dim = 42
hidden_dim = 512 #De 512 a 2000
layer_num = 3  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER POR LO MENOS 3
output_dim = 42

#model = BiRNN(input_dim, hidden_dim, layer_dim, output_dim)
#model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
model = LSTMModel(input_dim, hidden_dim, layer_num, output_dim)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#if torch.cuda.device_count() > 1:
#  print("Let's use", torch.cuda.device_count(), "GPUs!")
#  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#  model = nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))

print(torch.cuda.device_count())
model = nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))
model.to(device)
#model.train()

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.MSELoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 1.0e-04

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN THE MODEL
'''
# Number of steps to unroll
seq_dim = 99

iter = 0

#print("Epochs: "+str(num_epochs))
total_step = len(train_loader)
for epoch in range(num_epochs):

    totalLoss = 0

    for i, (frames, labels) in enumerate(train_loader):
        # Load images as tensors with gradient accumulation capabilites
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        frames = frames.view(frames.size()[0], seq_dim, input_dim).contiguous().requires_grad_().to(device)
        labels = labels.to(device)
        outputs = model(frames)
        outputs = outputs.view(frames.size()[0],seq_dim,14,3)

        loss = criterion(outputs, labels)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        #print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        totalLoss+=loss.item()

    print('Epoch [{}/{}]], Loss: {:.4f}'.format(epoch + 1, num_epochs, totalLoss/total_step))

torch.save(model.state_dict(),'models/weights.h5')

"""
        if iter % 100 == 0:
            # Calculate Loss
            loss = 0
            # Iterate through test dataset
            for (frames, labels) in test_loader:
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                frames = frames.view(-1, seq_dim, input_dim).requires_grad_().to(device)
                # Forward pass only to get logits/output
                outputs = model(frames)
                outputs1 = outputs.reshape(frames.shape[0], 14, 3)
                #print(outputs1)
                # Get predictions from the maximum value
                #_, predicted = torch.max(outputs1.data, 1)
                # Total number of labels
                loss += criterion(outputs1.cpu(), labels.cpu())
                #print(predicted)
                #print(outputs1.shape)
                #print(predicted.shape)
                #print(labels.shape)
                # Total correct predictions
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                #if torch.cuda.is_available():
                #    correct += (predicted.cpu() == labels.cpu()).sum()
                #else:
                #    correct += (predicted == outputs1).sum()
            #accuracy = 100 * correct / total
            loss = torch.sqrt(loss)
            # Print Loss
            print('Iteration: {}. Loss: {}'.format(iter, loss.item()))
"""