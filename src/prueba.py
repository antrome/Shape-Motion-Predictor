'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code
and in making it available on more platforms.


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]
- OpenCV [http://opencv.org/downloads.html]
  --> (alternatively: matplotlib [http://matplotlib.org/downloads.html])


About the Script:
=================
This script demonstrates loading the smpl model and rendering it using OpenDR
to render and OpenCV to display (or alternatively matplotlib can also be used
for display, as shown in commented code below).

This code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Create an OpenDR scene (with a basic renderer, camera & light)
  - Render the scene using OpenCV / matplotlib


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run
the hello world script now by typing the following:
>	python render_smpl.py


'''
from __future__ import print_function, division, absolute_import
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from src.smpl.smpl_webuser.serialization import load_model
import h5py
import cv2
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import imageio
from PIL import Image
import random
import src.utils.viz as viz
import torch

f=h5py.File("/home/finuxs/Shape-Motion-Predictor/datasets/h36m/h36train_x32_beta_pose.hdf5")

## Load SMPL model (here we load the female model)
m = load_model('/home/finuxs/Shape-Motion-Predictor/src/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

## Assign random pose and shape parametersqq
# m.pose[:] = np.random.rand(m.pose.size) * .2
# m.betas[:] = np.random.rand(m.betas.size) * .03

imgsPose=[]
imgsMov=[]
images=[]

# === Plot and animate ===
fig = plt.figure()
ax = plt.gca(projection='3d')
ob = viz.Ax3DPose(ax)

for i in range(1372):

    if i%100 == 0:
        m.pose[:] = f['000'].get('pose').value[1][i][0]
        movs = f['000'].get('x32').value[1][i]
        movs =torch.from_numpy(movs)
        print(m.pose[:])
        ## Create OpenDR renderer
        rn = ColoredRenderer()

        ## Assign attributes to renderer
        w, h = (640, 480)

        rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
        rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
        rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

        ## Construct point light source
        rn.vc = LambertianPointLight(
            f=m.f,
            v=rn.v,
            num_verts=len(m),
            light_pos=np.array([-1000,-1000,-2000]),
            vc=np.ones_like(m)*.9,
            light_color=np.array([1., 1., 1.]))

        # ## Show it using OpenCV
        imgsPose.append(rn.r)

        # Plot the conditioning ground truth
        ob.update(movs)
        plt.show(block=False)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.pause(0.01)
        imgsMov.append(data)

        """
        cv2.imshow('render_SMPL', rn.r)
        key = cv2.waitKey(500)  # pauses for 3 seconds before fetching next image
        if key == 27:  # if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break
        """

        ## Could also use matplotlib to display
        #import matplotlib.pyplot as plt
        #plt.ion()
        #plt.imshow(rn.r)
        #plt.show()
        #import pdb; pdb.set_trace()

# Put the predicted and gt together
for i in range(0, len(imgsMov)):
    print(imgsPose[i])
    print(imgsMov[i])
    images.append(np.hstack((imgsMov[i], imgsPose[i])))

imageio.mimsave("POSE.gif", imgsPose)
imageio.mimsave("MOV.gif", imgsMov)
