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


class H36M(DatasetBase):
    """ H36M dataset."""

    # Initialize your data, download, etc.
    def __init__(self, opt, is_for, subset, transform, dataset_type):
        super(H36M, self).__init__(opt, is_for, subset, transform, dataset_type)
        self._name = 'H36M'

        # init meta
        self._init_meta(opt)

        # normalize
        # self._normalize()

        # read dataset
        self._read_dataset()

        # read meta
        # self._read_meta()

        self.transform = transform

        # self._dataset_size = self.__len__()

    def _init_meta(self, opt):
        self._rgb = not opt[self._name]["use_bgr"]
        self._root = opt[self._name]["data_dir"]
        self._data_folder = opt[self._name]["data_folder"]
        self._meta_file = opt[self._name]["meta_file"]
        self._url = opt[self._name]["url"]
        self._tgz_md5 = opt[self._name]["tgz_md5"]
        self._data_dir = opt[self._name]["data_dir"]
        self._filename = opt[self._name]["filename"]

        if self._is_for == "train":
            self._ids_filename = self._opt[self._name]["train_ids_file"]
        elif self._is_for == "val":
            self._ids_filename = self._opt[self._name]["val_ids_file"]
        elif self._is_for == "test":
            self._ids_filename = self._opt[self._name]["test_ids_file"]
        else:
            raise ValueError(f"is_for={self._is_for} not valid")

    def _normalize(self):
        # read ids
        use_ids_filepath = os.path.join(self._root, self._ids_filename)
        valid_ids_root = self._read_valid_ids(use_ids_filepath)
        # load the picked numpy arrays
        data = []

        filepath = os.path.join("./"+self._data_dir, self._filename)
        with h5py.File(filepath, 'r') as f:

            for subseq in valid_ids_root:

                x_data = f['{:03d}'.format(int(subseq))][:]
                x_tensor = torch.from_numpy(x_data)

                data.append(x_tensor)

        tensors = torch.cat([i for i in data],1)

        self._mean = torch.mean(tensors,dim=(0,1,2),keepdim=True)
        self._std = torch.sqrt(torch.mean((tensors - self._mean)**2,dim=(0,1,2),keepdim=True))

        print(self._mean)
        print(self._std)

    def __getitem__(self,index):
        assert (index < self._dataset_size)

        data = self._data[index]

        # Pick a random camera
        cam = random.randint(0, 3)
        frames = random.randint(0, data.shape[1] - 101)
        x_data_cam_frame = data[cam][frames:frames + 99][:][:]

        labels_data_cam_frame = data[cam][frames + 1:frames + 100][:][:]
        self.tensor_x = torch.stack([torch.Tensor(i) for i in x_data_cam_frame])
        self.tensor_y = torch.stack([torch.Tensor(i) for i in labels_data_cam_frame])

        img, target = self.tensor_x, self.tensor_y

        # pack data
        sample = {'img': img, 'target': target}

        # apply transformations
        if self._transform is not None:
            sample = self._transform(sample)

        # reshape
        #self.tensor_x = self.tensor_x.reshape(self.tensor_x.size(0), self.tensor_x.size(1) * self.tensor_x.size(2))
        #self.tensor_y = self.tensor_y.reshape(self.tensor_y.size(0), self.tensor_y.size(1) * self.tensor_y.size(2))

        sample['img'] = sample['img'].reshape(sample['img'].size(0), sample['img'].size(1) * sample['img'].size(2))
        sample['target'] = sample['target'].reshape(sample['target'].size(0), sample['target'].size(1) * sample['target'].size(2))

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset(self):
        # read ids
        use_ids_filepath = os.path.join(self._root, self._ids_filename)
        valid_ids_root = self._read_valid_ids(use_ids_filepath)
        # load the picked numpy arrays
        self._data = []
        #self._targets = []

        filepath = os.path.join("./"+self._data_dir, self._filename)
        with h5py.File(filepath, 'r') as f:

            for subseq in valid_ids_root:

                x_data = f['{:03d}'.format(int(subseq))][:]
                x_tensor = torch.from_numpy(x_data)

                #Pick only one each 2 frames
                x_tensor = x_tensor[:,::2,:,:]

                #x_tensor = (x_tensor - self._mean)/self._std

                self._data.append(x_tensor)

        # dataset size
        self._dataset_size = len(self._data)

        """
        for file_name in valid_ids_root:
            file_path = os.path.join(self._root, self._data_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self._data.append(entry['data'])
                if 'labels' in entry:
                    self._targets.extend(entry['labels'])
                else:
                    self._targets.extend(entry['fine_labels'])

        # reshape data
        self._data = np.vstack(self._data).reshape(-1, 3, 32, 32)
        self._data = self._data.transpose((0, 2, 3, 1))  # convert to HWC
        """

    def _read_meta(self):
        path = os.path.join(self._root, self._data_folder, self._meta_file)
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data["label_names"]
        self._class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _read_valid_ids(self, file_path):
        ids = np.loadtxt(file_path, dtype=np.str)
        return np.expand_dims(ids, 0) if ids.ndim == 0 else ids

"""
    def _read_dataset(self):
        # read ids
        use_ids_filepath = os.path.join(self._root, self._ids_filename)
        valid_ids_root = self._read_valid_ids(use_ids_filepath)

        # load the picked numpy arrays
        self._data = []
        self._targets = []
        for file_name in valid_ids_root:
            file_path = os.path.join(self._root, self._data_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self._data.append(entry['data'])
                if 'labels' in entry:
                    self._targets.extend(entry['labels'])
                else:
                    self._targets.extend(entry['fine_labels'])

        # reshape data
        self._data = np.vstack(self._data).reshape(-1, 3, 32, 32)
        self._data = self._data.transpose((0, 2, 3, 1))  # convert to HWC

        # dataset size
        self._dataset_size = len(self._data)

    def _read_meta(self):
        path = os.path.join(self._root, self._data_folder, self._meta_file)
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data["label_names"]
        self._class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _read_valid_ids(self, file_path):
        ids = np.loadtxt(file_path, dtype=np.str)
        return np.expand_dims(ids, 0) if ids.ndim == 0 else ids

    def _download(self):
        # check already downloaded
        if os.path.isdir(os.path.join(self._root, self._data_folder)):
            return

        # download file
        print("It will take aprox 15 min...")
        download_url(self._url, self._root, self._filename, self._tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self._root, self._filename), "r:gz") as tar:
            tar.extractall(path=self._root)
"""

"""
class Human36MDataset(DatasetBase):
    def __init__(self, opt, is_for, subset, transform, dataset_type):
        super(Human36MDataset, self).__init__(opt, is_for, subset, transform, dataset_type)
        self._name = 'human36m'

        # init meta
        self._init_meta(opt)

        # download dataset if necessary
        #self._download()

        # read dataset
        self._read_dataset()

        # read meta
        self._read_meta()

    def _init_meta(self, opt):
        self._rgb = not opt[self._name]["use_bgr"]
        self._root = opt[self._name]["data_dir"]
        self._data_folder = opt[self._name]["data_folder"]
        self._meta_file = opt[self._name]["meta_file"]
        self._url = opt[self._name]["url"]
        self._filename = opt[self._name]["filename"]
        self._tgz_md5 = opt[self._name]["tgz_md5"]


        if self._is_for == "train":
            self._ids_filename = self._opt[self._name]["train_ids_file"]
        elif self._is_for == "val":
            self._ids_filename = self._opt[self._name]["val_ids_file"]
        elif self._is_for == "test":
            self._ids_filename = self._opt[self._name]["test_ids_file"]
        else:
            raise ValueError(f"is_for={self._is_for} not valid")

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        # get data
        img, target = self._data[index], self._targets[index]

        # pack data
        sample = {'img': img, 'target': target}

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
        self._targets = []
        for file_name in valid_ids_root:
            file_path = os.path.join(self._root, self._data_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self._data.append(entry['data'])
                if 'labels' in entry:
                    self._targets.extend(entry['labels'])
                else:
                    self._targets.extend(entry['fine_labels'])

        # reshape data
        self._data = np.vstack(self._data).reshape(-1, 3, 32, 32)
        self._data = self._data.transpose((0, 2, 3, 1))  # convert to HWC

        # dataset size
        self._dataset_size = len(self._data)

    def _read_meta(self):
        path = os.path.join(self._root, self._data_folder, self._meta_file)
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data["label_names"]
        self._class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _read_valid_ids(self, file_path):
        ids = np.loadtxt(file_path, dtype=np.str)
        return np.expand_dims(ids, 0) if ids.ndim == 0 else ids

    def _download(self):
        # check already downloaded
        if os.path.isdir(os.path.join(self._root, self._data_folder)):
            return

        # download file
        print("It will take aprox 15 min...")
        download_url(self._url, self._root, self._filename, self._tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self._root, self._filename), "r:gz") as tar:
            tar.extractall(path=self._root)
"""