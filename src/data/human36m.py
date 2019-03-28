# -*- coding: future_fstrings -*-
import os.path
from src.data.dataset import DatasetBase
import numpy as np
from torchvision.datasets.utils import download_url
import tarfile
import sys
import pickle
import h5py
import re
from os import walk
import glob
import scipy.io
import os, os.path
import sys
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
import copy


class H36M(DatasetBase):
    """ H36M dataset."""

    # Initialize your data, download, etc.
    def __init__(self, opt, is_for, subset, transform, dataset_type):
        super(H36M, self).__init__(opt, is_for, subset, transform, dataset_type)
        self._name = 'H36M'

        # init meta
        self._init_meta(opt)

        # Calculate Normalize Value
        # self._normalize()

        # read dataset
        self._read_dataset()

        # apply transforms
        self.transform = transform

    def _init_meta(self, opt):
        self._root = opt[self._name]["root"]
        self._filename = opt[self._name]["filename"]
        self._subsampling = opt[self._name]["subsampling"]
        self._seq_dim = opt["dataset"]["seq_dim"]

        if self._is_for == "train":
            self._ids_filename = self._opt[self._name]["train_ids_file"]
        elif self._is_for == "val":
            self._ids_filename = self._opt[self._name]["val_ids_file"]
        elif self._is_for == "test":
            self._ids_filename = self._opt[self._name]["test_ids_file"]
        else:
            raise ValueError(f"is_for={self._is_for} not valid")

    def __getitem__(self,index):
        assert (index < self._dataset_size)

        data = self._data[index]["pose"]
        databetas = self._data[index]["betas"]

        if not self._is_for == "test":

            # Pick a random camera
            cam = random.randint(0, 3)

            if self._subsampling == "window200":
                frames = random.randint(0, data.shape[1] - ((self._seq_dim+1)*2+1))
                x_data_cam_frame = np.zeros((self._seq_dim, data[cam][frames][:][:].shape[0], data[cam][frames][:][:].shape[1]))
                labels_data_cam_frame = np.zeros((self._seq_dim, data[cam][frames][:][:].shape[0], data[cam][frames][:][:].shape[1]))
                betas_frame = databetas[cam][0][:][:]

                for i in range(self._seq_dim):
                    # Pick a random frame for the subsampling
                    win = random.randint(0, 1)
                    x_data_cam_frame[i] = data[cam][frames + win][:][:]
                    labels_data_cam_frame[i] = data[cam][frames + win + 2][:][:]
                    frames = frames + 2
            elif self._subsampling == "window300":
                frames = random.randint(0, data.shape[1] - ((self._seq_dim+1)*3+1))
                x_data_cam_frame = np.zeros((self._seq_dim, data[cam][frames][:][:].shape[0], data[cam][frames][:][:].shape[1]))
                labels_data_cam_frame = np.zeros((self._seq_dim, data[cam][frames][:][:].shape[0], data[cam][frames][:][:].shape[1]))
                betas_frame = databetas[cam][0][:][:]

                for i in range(self._seq_dim):
                    # Pick a random frame for the subsampling
                    win = random.randint(0, 2)
                    x_data_cam_frame[i] = data[cam][frames + win][:][:]
                    labels_data_cam_frame[i] = data[cam][frames + win + 3][:][:]
                    frames = frames + 3
            elif self._subsampling == "odd":
                data = data[:, 1::2, :, :]
                frames = random.randint(0, data.shape[1] - (self._seq_dim+2))
                x_data_cam_frame = data[cam][frames:frames + self._seq_dim][:][:]
                labels_data_cam_frame = data[cam][frames + 1:frames + (self._seq_dim+1)][:][:]
                betas_frame = databetas[cam][0][:][:]
            #EvenSubSampling
            else:
                data = data[:, ::2, :, :]
                frames = random.randint(0, data.shape[1] - (self._seq_dim+2))
                x_data_cam_frame = data[cam][frames:frames + self._seq_dim][:][:]
                labels_data_cam_frame = data[cam][frames + 1:frames + (self._seq_dim+1)][:][:]
                betas_frame = databetas[cam][0][:][:]

            self.tensor_x = torch.stack([torch.Tensor(i) for i in x_data_cam_frame])
            self.tensor_y = torch.stack([torch.Tensor(i) for i in labels_data_cam_frame])
            self.betas = torch.stack([torch.Tensor(i) for i in betas_frame])
            img, target, betas = self.tensor_x, self.tensor_y, self.betas

            # pack data
            sample = {'img': img, 'target': target, 'betas': betas}

            # apply transformations
            if self._transform is not None:
                sample = self._transform(sample)

            #X32
            if sample['img'].size(1) == 32 and sample['img'].size(2) == 3:
                sample['img'] = sample['img'].reshape(sample['img'].size(0), sample['img'].size(1) * sample['img'].size(2))
                sample['target'] = sample['target'].reshape(sample['target'].size(0), sample['target'].size(1) * sample['target'].size(2))
            else:
                sample['img'] = sample['img'].reshape(sample['img'].size(0), sample['img'].size(2))
                sample['target'] = sample['target'].reshape(sample['target'].size(0), sample['target'].size(2))

        #testing
        else:
            x_data_cam_frame = data[0:self._seq_dim][:]
            labels_data_cam_frame = data[1:(self._seq_dim + 1)][:]
            betas_frame = databetas[0][:]

            self.tensor_x = torch.Tensor(x_data_cam_frame)
            self.tensor_y = torch.Tensor(labels_data_cam_frame)
            self.betas = torch.Tensor(betas_frame)
            img, target, betas = self.tensor_x, self.tensor_y, self.betas

            # pack data
            sample = {'img': img, 'target': target, 'betas': betas}

            # apply transformations
            if self._transform is not None:
                sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset(self):
        # read ids
        use_ids_filepath = os.path.join(self._root, self._ids_filename)
        valid_ids_root = self._read_valid_ids(use_ids_filepath)
        # load the picked numpy arrays
        self._data = []
        x_data = dict()
        x_tensor = dict()
        filepath = os.path.join("./"+self._root, self._filename)
        with h5py.File(filepath, 'r') as f:
            for subseq in valid_ids_root:
                #x_data["x32"] = f['{:03d}'.format(int(subseq))].get("x32")[()][:]
                x_data["pose"] = f['{:03d}'.format(int(subseq))].get("pose")[()][:]
                x_data["betas"] = f['{:03d}'.format(int(subseq))].get("betas")[()][:]
                #x_tensor["x32"] = torch.from_numpy(x_data["x32"]).float()
                x_tensor["pose"] = torch.from_numpy(x_data["pose"]).float()
                x_tensor["betas"] = torch.from_numpy(x_data["betas"]).float()

                self._data.append(copy.deepcopy(x_tensor))
                x_tensor.clear()

        # dataset size
        self._dataset_size = len(self._data)

    def _read_valid_ids(self, file_path):
        ids = np.loadtxt(file_path, dtype=np.str)
        return np.expand_dims(ids, 0) if ids.ndim == 0 else ids

    def _normalize(self):
        # read ids
        use_ids_filepath = os.path.join(self._root, self._ids_filename)
        valid_ids_root = self._read_valid_ids(use_ids_filepath)
        # load the picked numpy arrays
        data = []
        data1 = []
        data2 = []
        tensors = dict()
        x_data = dict()
        x_tensor = dict()
        filepath = os.path.join("./"+self._root, self._filename)
        with h5py.File(filepath, 'r') as f:

            for subseq in valid_ids_root:
                #x_data1 = f['{:03d}'.format(int(subseq))].get("x32")[()][:]
                x_data2 = f['{:03d}'.format(int(subseq))].get("pose")[()][:]
                #x_data["x32"] = f['{:03d}'.format(int(subseq))].get("x32")[()][:]
                x_data["betas"] = f['{:03d}'.format(int(subseq))].get("betas")[()][:]
                x_data["pose"] = f['{:03d}'.format(int(subseq))].get("pose")[()][:]
                #x_tensor1 = torch.from_numpy(x_data1)
                x_tensor2 = torch.from_numpy(x_data2)
                #x_tensor["x32"] = torch.from_numpy(x_data["x32"])
                x_tensor["betas"] = torch.from_numpy(x_data["betas"])
                x_tensor["pose"] = torch.from_numpy(x_data["pose"])

                #data1.append(x_tensor1)
                data.append(copy.deepcopy(x_tensor))
                data2.append(x_tensor2)

                x_tensor.clear()

            tensors1 = torch.cat([i for i in data1],1)
            self._mean = torch.mean(tensors1,dim=(0,1,2),keepdim=True)
            self._std = torch.sqrt(torch.mean((tensors1 - self._mean)**2,dim=(0,1,2),keepdim=True))
            print(self._mean)
            print(self._std)
            tensors2 = torch.cat([i for i in data2],1)
            self._mean = torch.mean(tensors2,dim=(0,1,2),keepdim=True)
            self._std = torch.sqrt(torch.mean((tensors2 - self._mean)**2,dim=(0,1,2),keepdim=True))
            print(self._mean)
            print(self._std)

        for key in data[0].keys():
            tensors[key] = torch.cat([i[key] for i in data],1)

            if key=="x32":
                self._mean = torch.mean(tensors[key],dim=(0,1,2),keepdim=True)
                self._std = torch.sqrt(torch.mean((tensors[key] - self._mean)**2,dim=(0,1,2),keepdim=True))
                print(self._mean)
                print(self._std)
            elif key=="pose":
                self._mean = torch.mean(tensors[key],dim=(0,1,2),keepdim=True)
                self._std = torch.sqrt(torch.mean((tensors[key] - self._mean)**2,dim=(0,1,2),keepdim=True))
                print(self._mean)
                print(self._std)
            elif key=="betas":
                self._mean = torch.mean(tensors[key],dim=(0,1,2,3),keepdim=True)
                self._std = torch.sqrt(torch.mean((tensors[key] - self._mean)**2,dim=(0,1,2,3),keepdim=True))
                print(self._mean)
                print(self._std)