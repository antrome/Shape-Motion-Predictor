import h5py
import numpy as np
import re
from os import walk
import glob
import scipy.io
import os, os.path
import sys
import copy
from src.utils import util, data_utils, pytorchangles, npangles
import torch
import math

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

#Define detections path, subjects, actions, subactions and cameras
FILE_PATH="/home/finuxs/Shape-Motion-Predictor/datasets/h36m/"
subjects = [[1,5,6,7,8], [9,11]] #11 Missing
actions = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
subactions = [1,2]
cameras = [1,2,3,4]

filepath=os.path.join(FILE_PATH, 'h36train_x32_beta_pose.hdf5')
f = h5py.File(filepath,'r')

cnt=0
#cnt=180

expmapnp = f["000"].get("pose").value[0][0][0]
expmaptorch = torch.from_numpy(f["000"].get("pose").value[0][0][0])
expmaptorch = expmaptorch.to(0)
estimRotMatnp = npangles.expmap_to_rotmat(expmapnp)
print(estimRotMatnp)
estimRotMattorch = pytorchangles.expmap_to_rotmat(expmaptorch,0)
print(estimRotMatnp)

estimEulernp = npangles.rotmat_to_euler(estimRotMatnp)
print(estimEulernp)
estimEulertorch = pytorchangles.rotmat_to_euler(estimRotMattorch,0)
print(estimEulertorch)

f.close()
