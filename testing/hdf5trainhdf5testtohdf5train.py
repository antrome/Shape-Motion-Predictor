import h5py
import numpy as np
import re
from os import walk
import glob
import scipy.io
import os, os.path
import sys
import copy

#Print the attributes of a hdf5 file
def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print ("    %s: %s" % (key, val))

#Define detections path, subjects, actions, subactions and cameras
SAVE_PATH="/media/finuxs/HDD/"
subjects = [[1,5,6,7,8], [9]]
actions = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
subactions = [1,2]
cameras = [1,2,3,4]

filepath=os.path.join(SAVE_PATH, 'h36train.hdf5')
filepath2=os.path.join(SAVE_PATH, 'h36test.hdf5')
filepath3=os.path.join(SAVE_PATH, 'h36trainval.hdf5')
f = h5py.File(filepath,'r')
g = h5py.File(filepath2,'r')
h = h5py.File(filepath3,'w')

cnt=0

for i in range(len(list(f.keys()))):
    tmp = f['{:03d}'.format(i)].value
    h['{:03d}'.format(cnt)] = tmp
    cnt=cnt+1

for i in range(len(list(g.keys()))):
    tmp = f['{:03d}'.format(i)].value
    h['{:03d}'.format(cnt)] = tmp
    cnt=cnt+1

print(list(h.keys()))
f.close()
g.close()
h.close()
