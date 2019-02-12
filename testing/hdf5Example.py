import h5py
import numpy as np
import re
from os import walk
import glob
import scipy.io

def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print ("    %s: %s" % (key, val))

#directory="/home/finuxs/TFM/H36M/H36M_skeleton/detections/S01/"

#for d in walk(directory):
#    print(d[0])

DETC_PATH="/home/finuxs/TFM/H36M/H36M_skeleton/detections/"
subjects = [1]
actions = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
subactions = [1,2]
cameras = [1,2,3,4]

for subject in subjects:
    for action in actions:
        for subaction in subactions:
            for camera in cameras:
                folder_name = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(subject, action, subaction, camera)
                print(folder_name)
                
filesnames = sorted(glob.iglob('/home/finuxs/TFM/H36M/H36M_skeleton/detections/S01/*14.', recursive=False))
print(filesnames)

for filename in sorted(glob.iglob('/home/finuxs/TFM/H36M/H36M_skeleton/detections/S01/**/*.mat', recursive=True)):
    print(filename)

nframes = sum(1 for _ in filesnames)
print(nframes)
cnts = np.zeros((1,4),dtype=np.int)
ncam=int(nframes/4)
u14s = np.zeros((1,15,2,4,ncam,14,2),dtype=np.float32)
u32s = np.zeros((1,15,2,4,ncam,32,2),dtype=np.float32)

f = h5py.File('h36.hdf5','w')

for filename in sorted(glob.iglob('/home/finuxs/TFM/H36M/H36M_skeleton/detections/S01/**/*.mat', recursive=True)):
    print(filename)
    cam=int(filename[-15:-14])-1
    subact=int(filename[-21:-20])-1
    act=int(filename[-21:-20])-1

    mat = scipy.io.loadmat(filename)
    #print(mat)
    mat3214 = {k: v for k, v in mat.items() if k.startswith('U14') or k.startswith('U32')}

    key1, val1 = next(iter(mat3214.items()))

    mat3214.pop(key1)

    key2, val2 = next(iter(mat3214.items()))

    print(cnts[0,cam])

    #print("    %s: %s" % (key1, val1))
    #print("    %s: %s" % (key2, val2))

    u14s[act,subact,cam,cnts[0,cam],:,:] = val1.astype(np.float32)
    u32s[act,subact,cam,cnts[0,cam],:,:] = val2.astype(np.float32)

    #print(u14s[cam,cnts[0,cam],:,:])

    cnts[0, cam] = cnts[0,cam]+1

print(u14s)
print(u32s)

for filename in sorted(glob.iglob('/home/finuxs/TFM/H36M/H36M_skeleton/detections/S01/Act02/Subact01/Cam01/*.mat', recursive=True)):
    print(filename)
    f[filename[:-19]+filename[-13:-4]+"/u14"] = u14s
    f[filename[:-19]+filename[-13:-4]+"/u32"] = u32s

f.visititems(print_attrs)
print(f["home/finuxs/TFM/H36M/H36M_skeleton/detections/S01/Act02/Subact01/frame1383/u14"])
f.close()

"""
temperature = np.random.random(1024)
dt = 10.0
start_time = 1375204299
station = 15
wind = np.random.random(2048)
dt_wind = 5.0

filename = 'weather.hdf5'
f = h5py.File(filename,'w')
#f = h5py.File(filename, 'r')
f["/train/frame1/skeleton"] = 1
f["/train/frame1/action"] = 2
f["/train/frame1/subject"] = 3
f["/train/frame1/subaction"] = 4

f["/train/frame2/skeleton"] = 5
f["/train/frame2/action"] = 6
f["/train/frame2/subject"] = 7
f["/train/frame2/subaction"] = 8

f["/train/frame1/"].visititems(print_attrs)

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])
print(data)
"""

"""
with h5py.File("weather.hdf5") as f:
	f["/15/temperature"] = temperature
	f["/15/temperature"].attrs["dt"] = 10.0
	f["/15/temperature"].attrs["start_time"] = 1375204299
	f["/15/wind"] = wind
	f["/15/wind"].attrs["dt"] = 5.0
	f.visititems(print_attrs)
	f.close()
"""
