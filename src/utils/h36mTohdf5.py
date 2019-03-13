import h5py
import numpy as np
import re
from os import walk
import glob
import scipy.io
import os, os.path
import sys

#Print the attributes of a hdf5 file
def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print ("    %s: %s" % (key, val))

#Define detections path, subjects, actions, subactions and cameras
DETC_PATH="/media/finuxs/HDD/H36M/H36M_skeleton/detections/"
SHAPE_PATH="/media/finuxs/HDD/H36M/H36M_shape/data/"
SAVE_PATH="/media/finuxs/HDD/"
subjects = [[1,5,6,7,8], [9]]
actions = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
subactions = [1,2]
cameras = [1,2,3,4]

filepath=os.path.join(SAVE_PATH, 'h36.hdf5')
f = h5py.File(filepath,'a')

for split in range(2):

    for subject in subjects[split]:
        print("Subject: "+str(subject))
        for action in actions:
            print("Action: "+str(action))
            for subaction in subactions:
                folder_name_no_cam = 'S{:02d}/Act{:02d}/Subact{:02d}/'.format(subject, action, subaction)
                nFrames = len([name for name in os.listdir(DETC_PATH+folder_name_no_cam+"Cam02/") if os.path.isfile(os.path.join(DETC_PATH+folder_name_no_cam+"Cam02/", name))])
                cnts = np.zeros((1, 4), dtype=np.int)
                u14s = np.zeros((len(cameras), nFrames, 14, 2), dtype=np.float32)
                u32s = np.zeros((len(cameras), nFrames, 32, 2), dtype=np.float32)
                x14s = np.zeros((len(cameras), nFrames, 14, 3), dtype=np.float32)
                x32s = np.zeros((len(cameras), nFrames, 32, 3), dtype=np.float32)
                ucpm14s = np.zeros((len(cameras), nFrames, 14, 2), dtype=np.float32)
                ucpm14scores = np.zeros((len(cameras), nFrames, 14, 1), dtype=np.float32)
                betass = np.zeros((len(cameras), nFrames, 1, 10), dtype=np.float32)
                posess = np.zeros((len(cameras), nFrames, 1, 72), dtype=np.float32)
                vertss = np.zeros((len(cameras), nFrames, 6890, 3), dtype=np.float32)
                for camera in cameras:
                    folder_name = 'S{:02d}/Act{:02d}/Subact{:02d}/Cam{:02d}/'.format(subject, action, subaction, camera)
                    annot_file = DETC_PATH + folder_name
                    cnt = 0

                    for frame in sorted(glob.iglob(annot_file+"*.mat")):

                        if(cnt % 1000) == 0:
                            print(frame)
                        cnt=cnt+1
                        folder_name_no_cam_frame = folder_name_no_cam + frame[-13:-4]
                        mat = scipy.io.loadmat(frame)

                        mat3214 = {k: v for k, v in mat.items() if k.startswith('U14') or k.startswith('U32')}
                        key1, val1 = next(iter(mat3214.items()))
                        mat3214.pop(key1)
                        key2, val2 = next(iter(mat3214.items()))
                        u14s[camera-1,cnts[0,camera-1],:,:] = val1.astype(np.float32)
                        u32s[camera-1,cnts[0,camera-1],:,:] = val2.astype(np.float32)

                        mat32142 = {k: v for k, v in mat.items() if k.startswith('X14') or k.startswith('X32')}
                        key1, val1 = next(iter(mat32142.items()))
                        mat32142.pop(key1)
                        key2, val2 = next(iter(mat32142.items()))
                        x14s[camera-1,cnts[0,camera-1],:,:] = val1.astype(np.float32)
                        x32s[camera-1,cnts[0,camera-1],:,:] = val2.astype(np.float32)

                        mat32143 = {k: v for k, v in mat.items() if k.startswith('Ucpm14')}
                        key1, val1 = next(iter(mat32143.items()))
                        mat32143.pop(key1)
                        key2, val2 = next(iter(mat32143.items()))
                        ucpm14s[camera-1,cnts[0,camera-1],:,:] = val1.astype(np.float32)
                        ucpm14scores[camera-1,cnts[0,camera-1],:,:] = val2.astype(np.float32)
                        cnts[0, camera-1] = cnts[0,camera-1]+1

                    nFrames = len([name for name in os.listdir(DETC_PATH + folder_name_no_cam + "Cam02/") if os.path.isfile(os.path.join(DETC_PATH + folder_name_no_cam + "Cam02/", name))])
                    cnts = np.zeros((1, 4), dtype=np.int)

                    annot_file = SHAPE_PATH + folder_name
                    cnt = 0

                    for frame in sorted(glob.iglob(annot_file+"*.mat")):

                        if(cnt % 1000) == 0:
                            print(frame)
                        cnt=cnt+1
                        folder_name_no_cam_frame = folder_name_no_cam + frame[-13:-4]
                        mat = scipy.io.loadmat(frame)

                        mat3214 = {k: v for k, v in mat.items() if k.startswith('betas')}
                        key1, val1 = next(iter(mat3214.items()))
                        betass[camera-1,cnts[0,camera-1],:,:] = val1.astype(np.float32)

                        mat32142 = {k: v for k, v in mat.items() if k.startswith('pose')}
                        key1, val1 = next(iter(mat32142.items()))
                        posess[camera-1,cnts[0,camera-1],:,:] = val1.astype(np.float32)

                        mat32143 = {k: v for k, v in mat.items() if k.startswith('verts')}
                        key1, val1 = next(iter(mat32143.items()))
                        vertss[camera-1,cnts[0,camera-1],:,:] = val1.astype(np.float32)
                        cnts[0, camera-1] = cnts[0,camera-1]+1


                if split == 0:
                    f["train/skeleton/"+folder_name_no_cam + "/u14"] = u14s
                    f["train/skeleton/"+folder_name_no_cam + "/u32"] = u32s
                    f["train/skeleton/"+folder_name_no_cam + "/x14"] = x14s
                    f["train/skeleton/"+folder_name_no_cam + "/x32"] = x32s
                    f["train/skeleton/"+folder_name_no_cam + "/ucpm14s"] = ucpm14s
                    f["train/skeleton/"+folder_name_no_cam + "/ucpm14scores"] = ucpm14scores
                    f["train/skeleton/"+folder_name_no_cam + "/betas"] = betass
                    f["train/skeleton/"+folder_name_no_cam + "/pose"] = posess
                    f["train/skeleton/"+folder_name_no_cam + "/verts"] = vertss
                else:
                    f["test/skeleton/"+folder_name_no_cam + "/u14"] = u14s
                    f["test/skeleton/"+folder_name_no_cam + "/u32"] = u32s
                    f["test/skeleton/"+folder_name_no_cam + "/x14"] = x14s
                    f["test/skeleton/"+folder_name_no_cam + "/x32"] = x32s
                    f["test/skeleton/"+folder_name_no_cam + "/ucpm14s"] = ucpm14s
                    f["test/skeleton/"+folder_name_no_cam + "/ucpm14scores"] = ucpm14scores
                    f["test/skeleton/"+folder_name_no_cam + "/betas"] = betass
                    f["test/skeleton/"+folder_name_no_cam + "/pose"] = posess
                    f["test/skeleton/"+folder_name_no_cam + "/verts"] = vertss


