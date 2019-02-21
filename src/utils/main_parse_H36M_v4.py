
import os
from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import cPickle as pickle
import math
from time import time
from glob import glob
import argparse

import cv2
import numpy as np
import chumpy as ch

import procrustes
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

import scipy.io as sio
from numpy import linalg as LA
import os
import socket
import copy



import sys
hostname = socket.gethostname()
if hostname == 'imacmore':
    sys.path.append('/Users/fmoreno/work/Research/39-HumanShape/SMPLify/smplify_public/smpl')
    plt.switch_backend("TkAgg")
    IMG_DIR = '/Users/fmoreno/work/Datasets/Human36M_VideoFrames/'
elif hostname == 'macmore':
    sys.path.append('/Users/francesc/work/Research/39-HumanShape/SMPLify/smplify_public/smpl')
    IMG_DIR = '/Users/Francesc/work/Datasets/Human36M_VideoFrames/'

from opendr.camera import ProjectPoints

from smpl_webuser.serialization import load_model
from smpl_webuser.verts import verts_decorated
from lib.sphere_collisions import SphereCollisions
from renderer import render_model
import matplotlib.pyplot as plt

from data_utils import define_paths
from data_utils import rotate_and_translate
from data_utils import project_points


#IMG_DIR = '../../04_Fit_SMPL_to_H36M/01_Generate_Data/Samples_H36M/images/'
MODEL_DIR = './models/'
DATA_SHAPE_DIR = '../neutrMosh/neutrSMPL_H3.6/'
DATA_2D_DETECT_DIR = '../../04_Fit_SMPL_to_H36M/01_Generate_Data/Samples_H36M/detections/'

SUBJECTS = {'7','8','9'} # {'1','5','6','7','8','9','11'}
ACTIONS = {'2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'}
SUBACTIONS = {'1','2'}
CAMERAS = {'1','2','3','4'}


SAVE_EVERY_X_IMAGES =10


cpm_to_lsp = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1, 0]
smpl_to_lsp = [8, 5,  2,  1, 4, 7, 21, 19, 17, 16, 18, 20, 12]



def main():
    for SUBJECT in SUBJECTS:
        for ACTION in ACTIONS:
            for SUBACTION in SUBACTIONS:
                for CAMERA in CAMERAS:
                    model_file, img_sequence_dir, data_shape_file, data_2D_detect_dir, output_dir_data, output_dir_img = define_paths(SUBJECT, ACTION, SUBACTION, CAMERA, IMG_DIR, MODEL_DIR, DATA_SHAPE_DIR, DATA_2D_DETECT_DIR)
                    #img_sequence_dir = '/Users/fmoreno/work/Datasets/Human36M_VideoFrames/S1/VideoFrames/Act11_Subact02_Cam01/'
                    #data_shape_file = '../neutrMosh/neutrSMPL_H3.6/S1/Smoking 1.pkl'

                    if not exists(output_dir_data):
                        makedirs(output_dir_data)
                        makedirs(output_dir_img)

                    # Load model
                    model = load_model(model_file)

                    # Load data
                    file = open(data_shape_file, 'rb')
                    contents = pickle.load(file)
                    file.close()
                    Poses = contents['poses']
                    Trans = contents['trans']
                    Betas = contents['betas']
                    nposes = Poses.shape[0]

                    # Load images
                    #image_files = os.listdir(img_sequence_dir)
                    image_files = [f for f in os.listdir(img_sequence_dir) if f.endswith('.jpg')]
                    nimages =len(image_files)

                    #Load 2D Detections
                    detect_2D_files = os.listdir(data_2D_detect_dir)
                    ndetect2D = len(detect_2D_files)

                    j = 0
                    CAM_ID = int(CAMERA)
                    for i in range(CAM_ID-1,nposes,4):

                        #iii = range(CAM_ID-1,nposes,4)
                        #j = 335
                        #i = iii[j]


                        #Generate output name
                        if j<=nimages-1:
                            output_file_mat = output_dir_data + image_files[j][:-4]+ '.mat'
                            output_file_jpg = output_dir_img + image_files[j][:-4]+ '.jpg'
                            img_name_full_path = img_sequence_dir + image_files[j]

                            if not exists(output_file_mat) and exists(img_name_full_path): #True

                                # Load image
                                img = cv2.imread(img_name_full_path)

                                # Pose and betas
                                pose = ch.array(Poses[i])
                                trans = ch.array(Trans[i])
                                shape = ch.array(Betas)

                                # Load 2D/3D joints and calibration matrices
                                detect_2d_full_path = data_2D_detect_dir + detect_2D_files[j]
                                contents = sio.loadmat(detect_2d_full_path)

                                U14 = np.array(contents['U14'], dtype='double')
                                A = contents['A']
                                R = contents['R']
                                T = contents['T']
                                X14 = np.array(contents['X14'], dtype='double')

                                X14proj, X14c = project_points(A,R,T,X14)
                                X14c_lsp = X14c[cpm_to_lsp,:]/1000   #convert from mm to meters
                                U14lsp = U14[cpm_to_lsp, :]


                                ###### Instantiate SMPL under the given pose  ######
                                trans = np.zeros(3)
                                pose[:3]=np.zeros(3)
                                sv = verts_decorated(
                                    trans=ch.array(trans),
                                    pose=ch.array(pose),
                                    v_template=model.v_template,
                                    J=model.J_regressor,
                                    betas=ch.array(shape),
                                    shapedirs=model.shapedirs,
                                    weights=model.weights,
                                    kintree_table=model.kintree_table,
                                    bs_style=model.bs_style,
                                    f=model.f,
                                    bs_type=model.bs_type,
                                    posedirs=model.posedirs,
                                    want_Jtr=True)

                                Xsmpl = sv.J_transformed
                                Xsmpl_lsp = sv.J_transformed.r[smpl_to_lsp,:]

                                ###### Align gt3d with the instantiated SMPL ######
                                _, Z, Rot_smpl_to_cam, b, translation_vector_smpl_to_cam = procrustes.compute_similarity_transform(
                                    X14c_lsp[:13, :], Xsmpl_lsp)
                                Xsmpl_lsp_align = Xsmpl_lsp.dot(Rot_smpl_to_cam) + translation_vector_smpl_to_cam

                                #we want the rotation matrix to premultiply, so we take the transpose
                                Rot_smpl_to_cam = Rot_smpl_to_cam.T
                                Xsmpl_lsp_align2 = (Rot_smpl_to_cam.dot(Xsmpl_lsp.T)).T + translation_vector_smpl_to_cam

                                rotation_vector, _ = cv2.Rodrigues(Rot_smpl_to_cam)
                                translation_vector = translation_vector_smpl_to_cam

                                # To instantiate model with specific R,T we need a slightly different translation vector (see mynotebook page 15)
                                j0 = Xsmpl[0, :].reshape(3, 1)
                                translation_vector2 = ch.array(translation_vector.reshape([3,1])) + Rot_smpl_to_cam.dot(j0) - j0

                                ##### Instantiate again with the estimated pose #####
                                pose[:3] = rotation_vector.T
                                sv2 = verts_decorated(
                                    trans=ch.array(translation_vector2),
                                    pose=ch.array(pose),
                                    v_template=model.v_template,
                                    J=model.J_regressor,
                                    betas=ch.array(shape),
                                    shapedirs=model.shapedirs,
                                    weights=model.weights,
                                    kintree_table=model.kintree_table,
                                    bs_style=model.bs_style,
                                    f=model.f,
                                    bs_type=model.bs_type,
                                    posedirs=model.posedirs,
                                    want_Jtr=True)

                                Xsmpl2 = sv2.J_transformed
                                Xsmpl2_lsp = Xsmpl2.r[smpl_to_lsp,:]

                                Ridentity=np.eye(3)
                                Tnull= np.zeros([3,1])
                                Xsmpl2_proj, _ = project_points(A,Ridentity,Tnull,Xsmpl2_lsp)

                                #### Plot Results  ####
                                h = img.shape[0]
                                w = img.shape[1]
                                rt = ch.array(rotation_vector.T)
                                rt = rt[0]
                                t = ch.array(translation_vector.T)
                                #t =t[0]

                                center = np.array([A[0,2],A[1,2]])
                                flength_u = A[0,0]
                                flength_v =A[1,1]
                                cam = ProjectPoints(
                                    f=np.array([flength_u, flength_v]), rt=rt, t=t, k=np.zeros(5), c=center)
                                # we are going to project the SMPL joints
                                cam.v = Xsmpl
                                Xsmpl_proj =cam

                                if j % SAVE_EVERY_X_IMAGES == 0:
                                    dist = np.abs(cam.t.r[2] - np.mean(sv.r, axis=0)[2])

                                    verts = sv.r
                                    im = (render_model(
                                        verts, model.f, w, h, cam, far=20 + dist) * 255.).astype('uint8')

                                    img_render_overlay = render_model(
                                        verts,
                                        model.f,
                                        w,
                                        h,
                                        cam,
                                        do_alpha=True,
                                        img=np.array(img),
                                        far=20 + dist,
                                        color_id=0)

                                    Img_render_overlay = (img_render_overlay * 255).astype('uint8')

                                    plt.subplot(131)
                                    vis_img = img.copy()
                                    plt.imshow(img[:, :, ::-1])
                                    for coord in np.around(U14).astype(int):
                                        if (coord[0] < vis_img.shape[1] and coord[0] >= 0 and
                                                    coord[1] < vis_img.shape[0] and coord[1] >= 0):
                                            cv2.circle(vis_img, tuple(coord), 7, [0, 255, 0],-1)
                                    for coord in np.around(X14proj).astype(int):
                                        if (coord[0] < vis_img.shape[1] and coord[0] >= 0 and
                                                    coord[1] < vis_img.shape[0] and coord[1] >= 0):
                                            cv2.circle(vis_img, tuple(coord), 10, [255, 0, 255])
                                    for coord in np.around(Xsmpl_proj).astype(int):
                                        if (coord[0] < vis_img.shape[1] and coord[0] >= 0 and
                                                    coord[1] < vis_img.shape[0] and coord[1] >= 0):
                                            cv2.circle(vis_img, tuple(coord), 15, [255, 0, 0])
                                    plt.imshow(vis_img[:, :, ::-1])

                                    plt.subplot(132)
                                    plt.cla()
                                    plt.imshow(im[:, :, ::-1])

                                    plt.subplot(133)
                                    plt.cla()
                                    vis_img = Img_render_overlay.copy()
                                    plt.imshow(Img_render_overlay)
                                    for coord in np.around(U14).astype(int):
                                        if (coord[0] < vis_img.shape[1] and coord[0] >= 0 and
                                                    coord[1] < vis_img.shape[0] and coord[1] >= 0):
                                            cv2.circle(vis_img, tuple(coord), 7, [255, 0, 255],-1)
                                    for coord in np.around(Xsmpl_proj).astype(int):
                                        if (coord[0] < vis_img.shape[1] and coord[0] >= 0 and
                                                    coord[1] < vis_img.shape[0] and coord[1] >= 0):
                                            cv2.circle(vis_img, tuple(coord), 15, [255, 0, 0])

                                    plt.imshow(vis_img[:, :, ::-1])

                                    # Save img
                                    plt.savefig(output_file_jpg)
                                    plt.clf()

                                # Save data: pose, shape,
                                DEBUG=0
                                if DEBUG == 0:
                                    sio.savemat(output_file_mat,
                                            {'verts': sv2.r,'trans': sv2.trans.r,'pose': sv2.pose.r, 'betas': sv2.betas.r,
                                             'Rot_smpl_to_cam':Rot_smpl_to_cam, 'translation_vector':translation_vector,
                                             'translation_vector2':translation_vector2.r, 'A':A, 'Xsmpl2_lsp':Xsmpl2_lsp,
                                             'Xsmpl2_proj': Xsmpl2_proj, 'U14':U14})
                                else:
                                    sio.savemat(output_file_mat,
                                            {'verts': sv2.r,'verts0':sv.r,'trans': sv2.trans.r,'pose': sv2.pose.r, 'betas': sv2.betas.r,
                                             'X14c_lsp': X14c_lsp, 'Xsmpl_lsp': Xsmpl_lsp, 'Xsmpl_proj':Xsmpl_proj.r,'A':A,
                                             'Xsmpl_lsp_align':Xsmpl_lsp_align,'Xsmpl_lsp_align2':Xsmpl_lsp_align2,
                                             'Xsmpl2_lsp':Xsmpl2_lsp, 'Xsmpl2_proj':Xsmpl2_proj, 'U14':U14 })




                        print (output_file_mat)
                        j = j+1



if __name__ == '__main__':


    main()
