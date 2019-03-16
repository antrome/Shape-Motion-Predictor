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
DETC_PATH="/media/finuxs/HDD/H36M/H36M_skeleton/detections/"
SHAPE_PATH="/media/finuxs/HDD/H36M/H36M_shape/data/"
FILE_PATH="/media/finuxs/HDD/"
SAVE_PATH="/home/finuxs/Shape-Motion-Predictor/datasets/h36m/"
subjects = [[], [11]] #11 Missing
actions = [3,4,5,7,8,9,10,11,12,13,14,15,16]
subactions = [1,2]
cameras = [1,2,3,4]

filepath=os.path.join(FILE_PATH, 'h36.hdf5')
filepath2=os.path.join(SAVE_PATH, 'h36train_x32_beta_pose.hdf5')
f = h5py.File(filepath,'r')
g = h5py.File(filepath2,'a')

#cnt=0
cnt=180

for split in range(2):

    for subject in subjects[split]:
        print("Subject: "+str(subject))
        for action in actions:
            print("Action: "+str(action))
            for subaction in subactions:
                folder_name_no_cam = 'S{:02d}/Act{:02d}/Subact{:02d}/'.format(subject, action, subaction)
                print(folder_name_no_cam)
                for camera in cameras:
                    folder_name = 'S{:02d}/Act{:02d}/Subact{:02d}/Cam{:02d}/'.format(subject, action, subaction, camera)
                if not (action==2 and subject==11):
                    if split == 0:
                        tmp = f["train/skeleton/"+folder_name_no_cam].get('x32').value
                        g['{:03d}'.format(cnt)+"/x32"]=tmp
                        tmp = f["train/skeleton/"+folder_name_no_cam].get('betas').value
                        g['{:03d}'.format(cnt)+"/betas"]=tmp
                        tmp = f["train/skeleton/"+folder_name_no_cam].get('pose').value
                        g['{:03d}'.format(cnt)+"/pose"]=tmp
                    else:
                        tmp = f["test/skeleton/"+folder_name_no_cam].get('x32').value
                        g['{:03d}'.format(cnt)+"/x32"]=tmp
                        tmp = f["test/skeleton/"+folder_name_no_cam].get('betas').value
                        g['{:03d}'.format(cnt)+"/betas"]=tmp
                        tmp = f["test/skeleton/"+folder_name_no_cam].get('pose').value
                        g['{:03d}'.format(cnt)+"/pose"]=tmp
                    cnt = cnt + 1

print(list(g.keys()))
f.close()
g.close()