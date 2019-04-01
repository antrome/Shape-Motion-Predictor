# -*- coding: future_fstrings -*-
from __future__ import print_function, division, absolute_import
import argparse
from src.options.config_parser import ConfigParser
from src.data.custom_dataset_data_loader import CustomDatasetDataLoader
from src.models.models import ModelsFactory
from src.utils.util import mkdir, tensor2im
from tqdm import tqdm
import time
import os
import cv2
from src.utils.wildUtils import readFrames, readFramesVideo
from src.hmr import demo
from src.options.config_parser import ConfigParser
from src.data.custom_dataset_data_loader import CustomDatasetDataLoader
from src.models.models import ModelsFactory
from src.utils.tb_visualizer import TBVisualizer
import torch
import torch.backends.cudnn as cudnn
from src.utils.util import append_dictionaries, mean_dictionary
import numpy as np
import time
import h5py
import os
import src.utils.viz as viz
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import imageio
from PIL import Image
import random
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
from PIL import Image,ImageFont,ImageDraw
import random
import src.utils.viz as viz
import torch
class Test:
    def __init__(self, args):
        config_parser = ConfigParser(set_master_gpu=False)
        self._opt = config_parser.get_config()
        self._opt["model"]["is_train"] = False

        # set output dir
        self._set_output()

        # add data
        # self._add_data()

        # prepare data
        self._prepare_data()

        # check options
        self._check_options()

        # Set master gpu
        self._set_gpus(args.gpu, config_parser)

        # create model
        model_type = self._opt["model"]["type"]
        self._model = ModelsFactory.get_by_name(model_type, self._opt)

        # test
        self._test_dataset()

    def _add_data(self):
        """
        cnt=206
        mkdir(os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames1"))
        readFrames(1,2,1,1,cnt,os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames1"))
        mkdir(os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames2"))
        readFrames(5,3,2,2,cnt+1,os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames2"))
        mkdir(os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames3"))
        readFrames(9,7,1,3,cnt+2,os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames3"))
        """
        cnt=208
        mkdir(os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames3"))
        readFramesVideo("walkDrunkFemale",cnt,os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames3"))
        """
        readFramesVideo("walkFemale",cnt,os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames4"))
        mkdir(os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames4"))
        readFramesVideo("walkFemale",cnt+1,os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames4"))
        mkdir(os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames5"))
        readFramesVideo("walkExhaustedMale",cnt+1,os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames5"))
        """

    def _prepare_data(self):
        data_loader_test = CustomDatasetDataLoader(self._opt, is_for="test")
        self._dataset_test = data_loader_test.load_data()
        self._dataset_test_size = len(data_loader_test)
        self._train_batch_size = data_loader_test.get_batch_size()
        print(f'#test images = {self._dataset_test_size}')

    def _check_options(self):
        assert self._opt["dataset_test"]["batch_size"] == 1
        assert self._opt["dataset_test"]["serial_batches"]

    def _set_gpus(self, gpu, config_parser):
        if gpu != -1:
            self._opt["misc"]["master_gpu"] = args.gpu
            self._opt["misc"]["G_gpus"] = [args.gpu]
        config_parser.set_gpus()

    def _set_output(self):
        self._save_folder = time.strftime("%d_%m_%H_%M_%S")
        mkdir(os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"]))

    def _test_dataset(self):
        self._model.set_eval()

        total_time = 0
        n_total_time = 0
        cnt=206
        for i_test_batch, test_batch in tqdm(enumerate(self._dataset_test), total=len(self._dataset_test)):
            # set inputs
            self._model.set_input(test_batch)

            moves_gt = dict()
            moves_predicted = dict()

            # get estimate
            start_wait = time.time()
            estimate = self._model.evaluate()
            moves = self._model.get_current_moves()
            moves_gt["moves_gt"] = moves["moves_gt"]
            moves_predicted["moves_predicted"] = moves["moves_predicted"]
            betas = self._model.get_current_betas()
            betas = betas["betas"]
            self._display_shape(moves_gt, moves_predicted, betas, 1, 1,
                                cnt,is_train=False)
            total_time += time.time() - start_wait
            n_total_time += 1
            cnt += 1
            # store estimate
            #self._save_img(estimate, i_test_batch)
        print(f"mean time per sample: {total_time/n_total_time}")

    def _save_img(self, img, id):
        filename = "{0:05d}.png".format(id)
        filepath = os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], self._save_folder, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)

    def _display_shape(self, gt_moves, predicted_moves, betas, dataset_size, batch_size, i_epoch, is_train):
        # Pick Up a Random Batch and Print it
        batch = random.randint(0, batch_size - 1)
        images_gt = []
        images_predicted = []
        images = []
        betasShow = betas[batch]
        betasShow = betasShow.reshape(betasShow.size(1))
        betasShow = betasShow.cpu().numpy()
        betasShow = list(betasShow)
        femaleBetas = [[-0.58770835, 0.3434618, 1.1994131, 0.6354899, -0.95589703,
                        0.6346986, -1.7816094, 1.1380256, 0.25826356, 2.1987565],
                       [-0.81385124, 0.51430994, -0.18375851, 0.33425307, 0.0014487166, 0.47330925, -1.3959571,
                        0.188319, 2.0626664, 1.5496577],
                       [-0.5000882, 0.31467184, 0.3889278, 0.48192567, -0.21293406, 0.7430237, 0.6407986, -0.3680312,
                        1.2822264, 0.13202368]]

        betasShowList =  [ '%.2f' % elem for elem in betasShow]

        for i in range(len(femaleBetas)):
            femaleBetas[i] = [ '%.2f' % elem for elem in femaleBetas[i]]

        if betasShowList in femaleBetas:
            m = load_model('src/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        #LOAD MALE MODEL
        else:
            m = load_model('src/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

        m.betas[:] = betasShow

        # === Plot and animate ===
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ob = viz.Ax3DPose(ax)
        for i in range(99):
            m.pose[:] = gt_moves["moves_gt"][0][i].detach().cpu().numpy()
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

            image = (rn.r * 255).round().astype(np.uint8)
            images_gt.append(image)

        for i in range(99):
            m.pose[:] = predicted_moves["moves_predicted"][0][i].detach().cpu().numpy()
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

            image = (rn.r * 255).round().astype(np.uint8)
            # ## Show it using OpenCV
            images_predicted.append(image)

        # Put the predicted and gt together
        for i in range(0, len(images_gt)):
            img = Image.fromarray(np.hstack((images_gt[i], images_predicted[i])))
            draw = ImageDraw.Draw(img)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            # draw.text((x, y),"Sample Text",(r,g,b))
            if i < len(images_gt)/2:
                draw.text((275, 0), "Ground Truth", (255, 255, 255))
                draw.text((925, 0), "Ground Truth", (255, 255, 255))
            else:
                draw.text((275, 0), "Ground Truth", (255, 255, 255))
                draw.text((925, 0), "Predicted", (255, 255, 255))

            images.append(img)

        imageio.mimsave(os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "test" + str(i_epoch) + ".gif"), images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to run test')
    args, _ = parser.parse_known_args()
    Test(args)
