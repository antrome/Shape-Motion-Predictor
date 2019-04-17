# -*- coding: future_fstrings -*-
from __future__ import print_function, division, absolute_import
import argparse
from src.utils.util import mkdir
from src.options.config_parser import ConfigParser
from src.data.custom_dataset_data_loader import CustomDatasetDataLoader
from src.models.models import ModelsFactory
from src.utils.util import append_dictionaries
import time
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from src.smpl.smpl_webuser.serialization import load_model
import cv2
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import imageio
from PIL import Image,ImageFont,ImageDraw
import random
import src.utils.viz as viz
class Test:
    def __init__(self, args):
        config_parser = ConfigParser(set_master_gpu=False)
        self._opt = config_parser.get_config()
        self._opt["model"]["is_train"] = False

        self._seq_dim = self._opt["networks"]["reg"]["hyper_params"]["seq_dim"]
        self._pred_dim = self._opt["networks"]["reg"]["hyper_params"]["pred_dim"]

        # set output dir
        self._set_output()

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
        val_errors = np.zeros((10,self._seq_dim-self._pred_dim))
        val_gt_moves = dict()
        val_gt_moves_aux = dict()
        val_predicted_moves = dict()
        val_predicted_moves_aux = dict()
        test_size = len(list(enumerate(self._dataset_test)))

        for i_test_batch, test_batch in enumerate(self._dataset_test):
            # set inputs
            self._model.set_input(test_batch)

            # get estimate
            estimate = self._model.evaluate()

            moves = self._model.get_current_moves()
            val_gt_moves_aux["moves_gt"] = moves["moves_gt"]
            val_predicted_moves_aux["moves_predicted"] = moves["moves_predicted"]
            val_gt_moves = append_dictionaries(val_gt_moves, val_gt_moves_aux)
            val_predicted_moves = append_dictionaries(val_predicted_moves, val_predicted_moves_aux)
            betas = self._model.get_current_betas()
            betas = betas['betas']

        self._display_shape(val_gt_moves, val_predicted_moves, betas, test_size, 1,
                            i_test_batch,is_train=False)

    def _save_img(self, img, id):
        filename = "{0:05d}.png".format(id)
        filepath = os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], self._save_folder, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)

    def _display_shape(self, gt_moves, predicted_moves, betas, dataset_size, batch_size, i_epoch, is_train):
        filepath = os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], "wildFrames")
        # Pick Up a Random Batch and Print it
        mov = random.randint(0, dataset_size - 1)
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

        #LOAD FEMALE MODEL
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
            m.pose[:] = gt_moves["moves_gt"][mov][batch][i].detach().cpu().numpy()
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
            m.pose[:] = predicted_moves["moves_predicted"][mov][batch][i].detach().cpu().numpy()
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
            images_predicted.append(image)

        # Put the predicted and gt together
        for i in range(0, len(images_gt)):
            img = Image.fromarray(np.hstack((images_gt[i], images_predicted[i])))
            draw = ImageDraw.Draw(img)
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