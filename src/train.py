from __future__ import print_function, division, absolute_import
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

class Train:
    def __init__(self):
        self._opt = ConfigParser().get_config()
        self._opt["model"]["is_train"] = True
        self._get_conf_params()
        cudnn.benchmark = True

        # create visualizer
        self._tb_visualizer = TBVisualizer(self._opt)

        # prepare data
        self._prepare_data()

        # check options
        self._check_options()

        # create model
        model_type = self._opt["model"]["type"]
        self._model = ModelsFactory.get_by_name(model_type, self._opt)

        # start train
        self._train()

    def _prepare_data(self):
        # create dataloaders
        data_loader_train = CustomDatasetDataLoader(self._opt, is_for="train")
        data_loader_val = CustomDatasetDataLoader(self._opt, is_for="val")

        # create dataset
        self._dataset_train = data_loader_train.load_data()
        self._dataset_val = data_loader_val.load_data()

        # get dataset properties
        self._dataset_train_size = len(data_loader_train)
        self._dataset_val_size = len(data_loader_val)
        self._num_batchesper_epoch_train = len(self._dataset_train)
        self._num_batchesper_epoch_val = len(self._dataset_val)

        # create visualizer
        self._tb_visualizer.print_msg('#train images = %d' % self._dataset_train_size)
        self._tb_visualizer.print_msg('#val images = %d' % self._dataset_val_size)

        # get batch size
        self._train_batch_size = data_loader_train.get_batch_size()
        self._val_batch_size = data_loader_val.get_batch_size()

    def _get_conf_params(self):
        self._load_epoch = self._opt["model"]["load_epoch"]
        self._nepochs_no_decay = self._opt["train"]["nepochs_no_decay"]
        self._nepochs_decay = self._opt["train"]["nepochs_decay"]
        self._print_freq_s = self._opt["logs"]["print_freq_s"]
        self._save_latest_freq_s = self._opt["logs"]["save_latest_freq_s"]
        self._display_freq_s = self._opt["logs"]["display_freq_s"]
        self._num_iters_validate = self._opt["train"]["num_iters_validate"]
        self._gifs_save_path = os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["gifs"])
        self._seq_dim = self._opt["dataset"]["seq_dim"]
        self._save_epoch = self._opt["logs"]["save_epoch"]
        self._print_epoch = self._opt["logs"]["print_epoch"]
        self._print_shape_epoch = self._opt["logs"]["print_shape_epoch"]

    def _check_options(self):
        assert self._opt["dataset_train"]["batch_size"] == self._opt["dataset_val"]["batch_size"], \
            "batch for val and train are required to be equal"

    def _train(self):
        # meta
        self._total_steps = self._load_epoch * self._dataset_train_size
        self._iters_per_epoch = len(self._dataset_train)
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()

        self._i_epoch = self._load_epoch + 1
        self._model.update_learning_rate(max(self._load_epoch + 1, 1))

        for i_epoch in range(self._load_epoch + 1, self._nepochs_no_decay + self._nepochs_decay + 1):
            self._i_epoch = i_epoch
            epoch_start_time = time.time()

            # train epoch visualizer
            if i_epoch % self._print_epoch == 0 or i_epoch == 1:
                self._train_epoch_vis(i_epoch)
                self._display_visualizer_avg_epoch(i_epoch)
            else:
                # train epoch no visualizer
                self._train_epoch(i_epoch)

            #save model
            if i_epoch % self._save_epoch == 0:
                self._model.save(i_epoch, "checkpoint")

            # update learning rate
            self._model.update_learning_rate(i_epoch + 1)

    def _train_epoch(self, i_epoch):
        self._model.set_train()

        for i_train_batch, train_batch in enumerate(self._dataset_train):
            do_visuals = False
            # train model
            self._model.set_input(train_batch)
            self._model.optimize_parameters(keep_data_for_visuals=do_visuals)
            self._total_steps += self._train_batch_size

            # save model
            # if self._last_save_latest_time is None or time.time() - self._last_save_latest_time > self._save_latest_freq_s:
            #    self._model.save(i_epoch, "checkpoint")
            #    self._last_save_latest_time = time.time()

    def _train_epoch_vis(self, i_epoch):
        self._epoch_train_e = dict()
        self._epoch_val_e = dict()
        self._epoch_train_mov = dict()
        self._epoch_val_mov = dict()

        # Print the movements
        self._display_visualizer_train(i_epoch, i_epoch)
        self._display_visualizer_val(i_epoch, i_epoch)

    def _display_terminal(self, iter_read_time, iter_procs_time, i_epoch, i_train_batch, visuals_flag):
        errors = self._model.get_current_errors()
        self._tb_visualizer.print_current_train_errors(i_epoch, i_train_batch, self._iters_per_epoch, errors,
                                                       iter_read_time, iter_procs_time, visuals_flag)

    """
    def _display_visualizer_train(self, total_steps, iter_read_time, iter_procs_time):
        #self._tb_visualizer.display_current_results(self._model.get_current_visuals(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_errors(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_scalars(), total_steps, is_train=True)
        self._tb_visualizer.plot_time(iter_read_time, iter_procs_time, total_steps)
        self._tb_visualizer.plot_histograms(self._model.get_current_histograms(), total_steps, is_train=True)
    """

    def _display_visualizer_avg_epoch(self, epoch):
        e_train = mean_dictionary(self._epoch_train_e)
        e_val = mean_dictionary(self._epoch_val_e)
        self._tb_visualizer.print_epoch_avg_errors(epoch, e_train, is_train=True)
        self._tb_visualizer.print_epoch_avg_errors(epoch, e_val, is_train=False)
        self._tb_visualizer.plot_scalars(e_train, epoch, is_train=True, is_mean=True)
        self._tb_visualizer.plot_scalars(e_val, epoch, is_train=False, is_mean=True)

    def _display_visualizer_train(self, i_epoch, total_steps):

        self._model.set_train()

        val_start_time = time.time()

        # evaluate self._opt.num_iters_validate epochs
        val_errors = dict()
        val_gt_moves = dict()
        val_gt_moves_aux = dict()
        val_predicted_moves = dict()
        val_predicted_moves_aux = dict()
        val_betas = torch.zeros(0,1,10,dtype=torch.float32)
        val_betas_aux = dict()
        train_size = len(list(enumerate(self._dataset_train)))
        do_visuals = False

        for i_val_batch, val_batch in enumerate(self._dataset_train):

            self._model.set_input(val_batch)
            self._model.optimize_parameters(keep_data_for_visuals=do_visuals)
            self._total_steps += self._train_batch_size

            if i_epoch % self._print_epoch == 0 or i_epoch == 1:
                # store errors
                errors = self._model.get_current_errors()
                val_errors = append_dictionaries(val_errors, errors)
                moves = self._model.get_current_moves()
                val_gt_moves_aux["moves_gt"] = moves["moves_gt"]
                val_predicted_moves_aux["moves_predicted"] = moves["moves_predicted"]
                val_gt_moves = append_dictionaries(val_gt_moves, val_gt_moves_aux)
                val_predicted_moves = append_dictionaries(val_predicted_moves, val_predicted_moves_aux)
                betas = self._model.get_current_betas()
                val_betas_aux = torch.stack([i.cpu() for i in betas["betas"]])
                val_betas = torch.cat((val_betas,val_betas_aux))

                # keep visuals
                if do_visuals:
                    self._tb_visualizer.display_current_results(self._model.get_current_visuals(), total_steps,
                                                                is_train=True)
                    self._tb_visualizer.plot_histograms(self._model.get_current_histograms(), total_steps,
                                                        is_train=True)
        # store error
        val_errors = mean_dictionary(val_errors)
        self._epoch_train_e = append_dictionaries(self._epoch_train_e, val_errors)
        # Print the movements
        #self._display_movements(val_gt_moves, val_predicted_moves, val_size, i_epoch, is_train=True)
        # Print the shape
        if i_epoch % self._print_shape_epoch == 0:
            self._display_shape(val_gt_moves, val_predicted_moves, val_betas, train_size, self._train_batch_size,i_epoch, is_train=True)
        # visualize
        t = (time.time() - val_start_time)
        self._tb_visualizer.plot_scalars(val_errors, total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_scalars(), total_steps, is_train=True)

    def _display_visualizer_val(self, i_epoch, total_steps):
        val_start_time = time.time()

        # set model to eval
        self._model.set_eval()
        # evaluate self._opt.num_iters_validate epochs
        val_errors = dict()
        val_gt_moves = dict()
        val_gt_moves_aux = dict()
        val_predicted_moves = dict()
        val_predicted_moves_aux = dict()
        val_betas = torch.zeros(0,1,10,dtype=torch.float32)
        val_betas_aux = dict()
        val_size = len(list(enumerate(self._dataset_val)))
        with torch.no_grad():
            #vis_batch_idx = np.random.randint(min(self._num_iters_validate, self._dataset_val_size))
            for i_val_batch, val_batch in enumerate(self._dataset_val):
                if i_val_batch == self._num_iters_validate:
                    break
                # evaluate model
                # keep_data_for_visuals = (i_val_batch == vis_batch_idx)
                keep_data_for_visuals = False
                self._model.set_input(val_batch)
                self._model.forward(keep_data_for_visuals=keep_data_for_visuals)

                # store errors
                errors = self._model.get_current_errors()
                val_errors = append_dictionaries(val_errors, errors)
                moves = self._model.get_current_moves()
                val_gt_moves_aux["moves_gt"] = moves["moves_gt"]
                val_predicted_moves_aux["moves_predicted"] = moves["moves_predicted"]
                val_gt_moves = append_dictionaries(val_gt_moves, val_gt_moves_aux)
                val_predicted_moves = append_dictionaries(val_predicted_moves, val_predicted_moves_aux)
                betas = self._model.get_current_betas()
                val_betas_aux = torch.stack([i.cpu() for i in betas["betas"]])
                val_betas = torch.cat((val_betas,val_betas_aux))

                # keep visuals
                if keep_data_for_visuals:
                    self._tb_visualizer.display_current_results(self._model.get_current_visuals(), total_steps,
                                                                is_train=False)
                    self._tb_visualizer.plot_histograms(self._model.get_current_histograms(), total_steps,
                                                        is_train=False)
            # store error
            val_errors = mean_dictionary(val_errors)
            self._epoch_val_e = append_dictionaries(self._epoch_val_e, val_errors)
            # Print the movements
            #self._display_movements_val(val_gt_moves, val_predicted_moves, val_size, i_epoch, is_train=False)
            # Print the shape
            if i_epoch % self._print_shape_epoch == 0:
                self._display_shape(val_gt_moves, val_predicted_moves, val_betas, val_size, self._val_batch_size, i_epoch, is_train=False)

        # visualize
        t = (time.time() - val_start_time)
        # self._tb_visualizer.print_current_validate_errors(i_epoch, val_errors, t)
        self._tb_visualizer.plot_scalars(val_errors, total_steps, is_train=False)

        # set model back to train
        self._model.set_train()

    def _display_movements(self, gt_moves, predicted_moves, dataset_size, i_epoch, is_train):
        # Pick Up a Random Batch and Print it

        if is_train:
            mov = random.randint(0, self._num_batchesper_epoch_train - 1)
        else:
            mov = random.randint(0, self._num_batchesper_epoch_val - 1)

        batch = random.randint(0, dataset_size) - 1
        images_gt = []
        images_predicted = []
        images = []

        # === Plot and animate ===
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ob = viz.Ax3DPose(ax)

        # Plot the conditioning ground truth
        for i in range(self._seq_dim):
            ob.update(gt_moves["moves_gt"][mov][batch][i, :].detach())
            plt.show(block=False)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.pause(0.01)
            images_gt.append(data)

        plt.close(fig)

        # Plot the conditioning predicted
        for i in range(self._seq_dim):
            ob.update(predicted_moves["moves_predicted"][mov][batch][i, :].detach(), lcolor="#9b59b6", rcolor="#2ecc71")
            plt.show(block=False)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.pause(0.01)
            images_predicted.append(data)

        plt.close(fig)

        # Put the predicted and gt together
        for i in range(0, len(images_gt)):
            images.append(np.hstack((images_gt[i], images_predicted[i])))

        if is_train:
            imageio.mimsave(os.path.join(self._gifs_save_path, "train", "epoch" + str(i_epoch) + ".gif"), images)
        else:
            imageio.mimsave(os.path.join(self._gifs_save_path, "val", "epoch" + str(i_epoch) + ".gif"), images)

    def _display_shape(self, gt_moves, predicted_moves, betas, dataset_size, batch_size, i_epoch, is_train):
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

        if betasShowList in femaleBetas:
            m = load_model('src/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        #LOAD MALE MODEL
        else:
            m = load_model('src/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

        m.betas[:] = betasShow

        """
        for i in range(int(len(predicted_moves)/2)-5,int(len(predicted_moves)/2)+5):
            predicted_moves["moves_predicted"][mov][batch][i,:] = (predicted_moves["moves_predicted"][mov][batch][i,:]+predicted_moves["moves_predicted"][mov][batch][i+1,:])/2
        """

        # === Plot and animate ===
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ob = viz.Ax3DPose(ax)
        for i in range(self._seq_dim):
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

        for i in range(self._seq_dim):
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

        if is_train:
            imageio.mimsave(os.path.join(self._gifs_save_path, "train", "epoch" + str(i_epoch) + ".gif"), images)
        else:
            imageio.mimsave(os.path.join(self._gifs_save_path, "val", "epoch" + str(i_epoch) + ".gif"), images)

if __name__ == "__main__":
    Train()
