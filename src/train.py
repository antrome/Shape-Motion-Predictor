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
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import imageio
from PIL import Image
import random


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
        self._num_batchesper_epoch = len(self._dataset_train)

        # print(self._dataset_train_size)
        # print(self._dataset_val_size)
        # print(len(list(enumerate(self._dataset_train))))
        # print(len(list(enumerate(self._dataset_val))))

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
            if i_epoch % 20 == 0 or i_epoch == 1:
                self._train_epoch_vis(i_epoch)
                self._display_visualizer_avg_epoch(i_epoch)
            else:
                # train epoch no visualizer
                self._train_epoch(i_epoch)

            """
            if i_epoch%100 == 0:
                self._model.save(i_epoch, "checkpoint")

                # print epoch info
                time_epoch = time.time() - epoch_start_time
                self._tb_visualizer.print_msg('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                      (i_epoch, self._nepochs_no_decay + self._nepochs_decay, time_epoch,
                       time_epoch / 60, time_epoch / 3600))

                # print epoch error
                self._display_visualizer_avg_epoch(i_epoch)
            """
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
        val_size = len(list(enumerate(self._dataset_val)))
        do_visuals = False

        for i_val_batch, val_batch in enumerate(self._dataset_train):

            self._model.set_input(val_batch)
            self._model.optimize_parameters(keep_data_for_visuals=do_visuals)
            self._total_steps += self._train_batch_size

            if i_epoch % 20 == 0 or i_epoch == 1:
                # store errors
                errors = self._model.get_current_errors()
                val_errors = append_dictionaries(val_errors, errors)
                moves = self._model.get_current_moves()
                val_gt_moves_aux["moves_gt"] = moves["moves_gt"]
                val_predicted_moves_aux["moves_predicted"] = moves["moves_predicted"]
                val_gt_moves = append_dictionaries(val_gt_moves, val_gt_moves_aux)
                val_predicted_moves = append_dictionaries(val_predicted_moves, val_predicted_moves_aux)

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
        self._display_movements_train(val_gt_moves, val_predicted_moves, val_size, i_epoch)

        # visualize
        t = (time.time() - val_start_time)
        # self._tb_visualizer.print_current_validate_errors(i_epoch, val_errors, t)
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
        val_size = len(list(enumerate(self._dataset_val)))
        with torch.no_grad():
            vis_batch_idx = np.random.randint(min(self._num_iters_validate, self._dataset_val_size))
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
            self._display_movements_val(val_gt_moves, val_predicted_moves, val_size, i_epoch)

        # visualize
        t = (time.time() - val_start_time)
        # self._tb_visualizer.print_current_validate_errors(i_epoch, val_errors, t)
        self._tb_visualizer.plot_scalars(val_errors, total_steps, is_train=False)

        # set model back to train
        self._model.set_train()

    def _display_movements_train(self, gt_moves, predicted_moves, dataset_size, i_epoch):
        # Pick Up a Random Batch and Print it
        batch = random.randint(0, dataset_size) - 1
        images_gt = []
        images_predicted = []
        images = []

        # === Plot and animate ===
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ob = viz.Ax3DPose(ax)

        # Plot the conditioning ground truth
        for i in range(99):
            ob.update(gt_moves["moves_gt"][batch][i, :].detach())
            plt.show(block=False)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.pause(0.01)
            images_gt.append(data)

        # Plot the conditioning predicted
        for i in range(99):
            ob.update(predicted_moves["moves_predicted"][batch][i, :].detach(), lcolor="#9b59b6", rcolor="#2ecc71")
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

        imageio.mimsave(os.path.join(self._gifs_save_path, "train", "epoch" + str(i_epoch) + ".gif"), images)

    def _display_movements_val(self, gt_moves, predicted_moves, dataset_size, i_epoch):
        # Pick Up a Random Batch and Print it
        batch = random.randint(0, dataset_size) - 1
        images_gt = []
        images_predicted = []
        images = []

        # === Plot and animate ===
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ob = viz.Ax3DPose(ax)

        # Plot the conditioning ground truth
        for i in range(99):
            ob.update(gt_moves["moves_gt"][batch][i, :])
            plt.show(block=False)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.pause(0.01)
            images_gt.append(data)

        # Plot the conditioning predicted
        for i in range(99):
            ob.update(predicted_moves["moves_predicted"][batch][i, :], lcolor="#9b59b6", rcolor="#2ecc71")
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

        imageio.mimsave(os.path.join(self._gifs_save_path, "val", "epoch" + str(i_epoch) + ".gif"), images)


if __name__ == "__main__":
    Train()
