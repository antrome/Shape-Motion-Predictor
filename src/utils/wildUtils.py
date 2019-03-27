import h5py
import numpy as np
import re
from os import walk
import glob
import scipy.io
import os, os.path
import sys
import cv2
import random
from src.hmr import demo

def readFrames(subject,action,subaction,camera,savefolder):
    #Define detections path, subjects, actions, subactions and cameras
    IMAGE_PATH="/media/finuxs/HDD/H36M/H36M_skeleton/images/"
    SAVE_PATH = "/home/finuxs/Shape-Motion-Predictor/datasets/h36m/"
    subjects = subject
    actions = action
    subactions = subaction
    cameras = camera

    filepath = os.path.join(SAVE_PATH, 'h36train_x32_beta_pose.hdf5')
    g = h5py.File(filepath, 'a')

    folder_name = 'S{:02d}/Act{:02d}/Subact{:02d}/Cam{:02d}/'.format(subjects, actions, subactions, cameras)
    annot_file = IMAGE_PATH + folder_name
    j=0
    length = len(sorted(glob.iglob(annot_file+"*.jpg")))
    i=random.randint(0,length-100)
    for frame in sorted(glob.iglob(annot_file+"*.jpg")):
        j += 1
        if j >= i and j<i+100:
            #print(frame)
            fr=cv2.imread(frame, 1)
            f=j+i
            cv2.imwrite("./"+savefolder+"/frame%d.jpg" % f, fr)  # save frame as JPEG file

    cnt=206
    shapes, betas = demo.main(savefolder)
    g['{:03d}'.format(cnt) + "/betas"] = betas
    g['{:03d}'.format(cnt) + "/pose"] = shapes
    print(list(g.keys()))
    g.close()

