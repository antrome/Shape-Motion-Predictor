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

def readFrames(subject,action,subaction,camera,cnt,savefolder):
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

    print(list(g.keys()))
    shapes, betas = demo.main(savefolder)
    g['{:03d}'.format(cnt) + "/betas"] = betas
    g['{:03d}'.format(cnt) + "/pose"] = shapes
    g.close()

    """
    for frame in sorted(glob.iglob("./"+savefolder+"/"+"*.jpg")):
        os.remove(frame)
    os.rmdir("./"+savefolder+"/")
    """

def readFramesVideo(video,cnt,savefolder):
    #Define detections path, subjects, actions, subactions and cameras
    SAVE_PATH = "/home/finuxs/Shape-Motion-Predictor/datasets/h36m/"
    VIDEO_PATH = "/home/finuxs/Shape-Motion-Predictor/datasets/endlessReference/"+video+"/"+video+".mp4"
    filepath = os.path.join(SAVE_PATH, 'h36train_x32_beta_pose.hdf5')
    g = h5py.File(filepath, 'a')
    j=0

    vidcap = cv2.VideoCapture(VIDEO_PATH)
    print(VIDEO_PATH)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = random.randint(0, length - 100)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
    success, image = vidcap.read()
    count = 0
    print(type(count))
    print(type(i))
    print(savefolder)

    while success:
        f = count + i
        success, image = vidcap.read()
        crop_img = image[:, 0:700]
        """
        cv2.imshow("cropped", crop_img)
        cv2.waitKey(0)
        """

        cv2.imwrite("./" + savefolder + "/frame%d.jpg" % f, crop_img)  # save frame as JPEG file
        count += 1

        if count == 100:
            break

    print(list(g.keys()))
    shapes, betas = demo.main(savefolder)
    g['{:03d}'.format(cnt) + "/betas"] = betas
    g['{:03d}'.format(cnt) + "/pose"] = shapes
    g.close()

    """
    for frame in sorted(glob.iglob("./"+savefolder+"/"+"*.jpg")):
        os.remove(frame)
    os.rmdir("./"+savefolder+"/")
    """


import cv2

if __name__ == '__main__':

    video = cv2.VideoCapture("/home/finuxs/Shape-Motion-Predictor/datasets/endlessReference/walkExhaustedMale/walkExhaustedMale.mp4");

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    video.release();