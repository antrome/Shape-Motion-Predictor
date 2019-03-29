import torch
from collections import OrderedDict
from src.utils import util, data_utils, pytorchangles
from .models import BaseModel
from src.networks.networks import NetworksFactory
from src.utils.plots import plot_estim
import numpy as np
import random

class ResLstm(BaseModel):
    def __init__(self, opt):
        super(ResLstm, self).__init__(opt)
        self._name = 'lstm'

        # init input params
        self._init_set_input_params()

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt["model"]["load_epoch"] > 0:
            self.load()

        # init losses
        if self._is_train:
            self._init_losses()

        # prefetch inputs
        self._init_prefetch_inputs(opt)

    def _init_set_input_params(self):
        self._B = self._opt[self._dataset_type]["batch_size"]           # batch
        self._Id = self._opt[self._dataset_type]["input_dim"]           # input dimension
        self._Idr = self._opt[self._dataset_type]["input_rows"]         # input dimension rows
        self._Idc = self._opt[self._dataset_type]["input_cols"]         # input dimension cols
        self._Hd = self._opt[self._dataset_type]["hidden_dim"]          # hidden dimension
        self._Ld = self._opt[self._dataset_type]["layer_dim"]           # layer dimension
        self._Od = self._opt[self._dataset_type]["output_dim"]          # output dimension
        self._Sd = self._opt[self._dataset_type]["seq_dim"]             # sequence dimension
        self._optim = self._opt["train"]["optim"]                       # optimizer used for the training
        self._inputType = self._opt["networks"]["reg"]["input_type"]    # input type of the network
        self._loss_type = self._opt["train"]["loss"]                    # loss type
        self._weight_decay = self._opt["train"]["weight_decay"]         # weight decay
        self._dropout = self._opt["train"]["dropout"]                   # dropout

    def _init_create_networks(self):
        # create reg
        reg_type = self._opt["networks"]["reg"]["type"]
        reg_hyper_params = self._opt["networks"]["reg"]["hyper_params"]
        self._reg = NetworksFactory.get_by_name(reg_type, **reg_hyper_params)
        self._reg = torch.nn.DataParallel(self._reg, device_ids=self._reg_gpus_ids)

    def _init_train_vars(self):
        self._current_lr = self._opt["train"]["reg_lr"]

        if self._optim == "adam":
            self._optimizer = torch.optim.Adam(self._reg.parameters(), lr=self._current_lr,
                                               weight_decay=self._weight_decay)
        elif self._optim == "adamax":
            self._optimizer = torch.optim.Adamax(self._reg.parameters(), lr=self._current_lr,
                                                 weight_decay=self._weight_decay)
        elif self._optim == "adadelta":
            self._optimizer = torch.optim.Adadelta(self._reg.parameters(), lr=self._current_lr,
                                                   weight_decay=self._weight_decay)
        elif self._optim == "adagrad":
            self._optimizer = torch.optim.Adagrad(self._reg.parameters(), lr=self._current_lr,
                                                  weight_decay=self._weight_decay)
        else:#Default SGD
            self._optimizer = torch.optim.SGD(self._reg.parameters(), lr=self._current_lr,
                                              weight_decay=self._weight_decay)

    def _init_losses(self):
        self._criterion = torch.nn.MSELoss().to(self._device_master)

    def _init_prefetch_inputs(self,opt):
        self._input_img = torch.zeros([self._B, self._Sd, self._Idr*self._Idc]).to(self._device_master)
        self._input_target = torch.zeros([self._B, self._Sd, self._Idr*self._Idc], dtype=torch.float32).to(self._device_master)
        self._betas1 = torch.zeros([self._B, 1, 10], dtype=torch.float32).to(self._device_master)
        self._global_translation_target = torch.zeros([self._B, self._Sd, self._Idr*self._Idc], dtype=torch.float32).to(self._device_master)
        self._global_translation_img = torch.zeros([self._B, self._Sd, self._Idr*self._Idc], dtype=torch.float32).to(self._device_master)
        self._input_target_zeros = torch.zeros([self._B, self._Sd, self._Idr*self._Idc], dtype=torch.float32).to(self._device_master)
        self._input_img_zeros = torch.zeros([self._B, self._Sd, self._Idr*self._Idc], dtype=torch.float32).to(self._device_master)
        #opt for unormalize
        self._mean = torch.FloatTensor(opt["transforms"]["normalize"]["general_args"]["mean"]).to(self._device_master)
        self._std = torch.FloatTensor(opt["transforms"]["normalize"]["general_args"]["std"]).to(self._device_master)

    def set_input(self, input):
        #Get Betas
        self._betas1.copy_(input['betas'])
        self._input_img.copy_(input['img'])
        #Defauly Input Type
        self._input_target.copy_(input['target'])
        self._input_target_zeros.copy_(input['target'])
        self._input_img_zeros.copy_(input['img'])
        self._global_translation_target.copy_(input['target'])
        self._global_translation_img.copy_(input['img'])

        #check input type
        if self._inputType == "hidden50FramesInput69par":
            self._input_img[:, 50:99, 3:] = torch.zeros(
                self._input_img[:, 50:99, 3:].size(0),
                self._input_img[:, 50:99, 3:].size(1),
                self._input_img[:, 50:99, 3:].size(2), dtype=torch.float32)
        elif self._inputType == "hidden50FramesInput66par":
            self._input_img[:, 50:99, 6:] = torch.zeros(
                self._input_img[:, 50:99, 6:].size(0),
                self._input_img[:, 50:99, 6:].size(1),
                self._input_img[:, 50:99, 6:].size(2), dtype=torch.float32)
        elif self._inputType == "repeatedFrame50Input69par":
            temp_tensor = torch.zeros(self._input_img[:, 50, 3:].size(0),49,self._input_img[:, 50, 3:].size(1), dtype=torch.float32)
            temp_tensor = torch.stack([self._input_img[:, 50, 3:] for i in range(49)],dim=1)
            self._input_img[:, 50:, 3:] = temp_tensor
        elif self._inputType == "repeatedFrame50Input66par":
            temp_tensor = torch.zeros(self._input_img[:, 50, 6:].size(0),49,self._input_img[:, 50, 6:].size(1), dtype=torch.float32)
            temp_tensor = torch.stack([self._input_img[:, 50, 6:] for i in range(49)],dim=1)
            self._input_img[:, 50:, 6:] = temp_tensor
        # move to gpu
        self._input_img = self._input_img.to(self._device_master)
        self._input_target = self._input_target.to(self._device_master)
        self._betas1 = self._betas1.to(self._device_master)
        self._global_translation_target = self._global_translation_target.to(self._device_master)
        self._global_translation_img = self._global_translation_img.to(self._device_master)

    def set_train(self):
        self._reg.train()
        self._is_train = True

    def set_eval(self):
        self._reg.eval()
        self._is_train = False

    def evaluate(self):
        # set model to eval
        is_train = self._is_train
        if is_train:
            self.set_eval()

        # estimate object categories
        with torch.no_grad():
            eval = self.forward(keep_data_for_visuals=False, estimate_loss=False)
            #eval = np.transpose(self._vis_input_img, (1, 2, 0))

        # set model back to train if necessary
        if is_train:
            self.set_train()

        return eval

    def optimize_parameters(self, keep_data_for_visuals=False):
        keep_data_for_visuals=False
        if self._is_train:
            # calculate loss
            loss = self.forward(keep_data_for_visuals=keep_data_for_visuals)

            # optimize
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        else:
            raise ValueError('Trying to optimize in non-training mode!')

    def forward(self, keep_data_for_visuals=False, estimate_loss=True):
        # generate img
        self._input_img[:, :, 0:3] = torch.zeros(
            self._input_img[:, :, 0:3].size(0),
            self._input_img[:, :, 0:3].size(1),
            self._input_img[:, :, 0:3].size(2), dtype=torch.float32)


        self._input_target[:, :, 0:3] = torch.zeros(
            self._input_target[:, :, 0:3].size(0),
            self._input_target[:, :, 0:3].size(1),
            self._input_target[:, :, 0:3].size(2), dtype=torch.float32)

        estimf, estimi = self._estimate(self._input_img)

        # estimate loss
        if estimate_loss:
            self._loss_gtf = self._criterion(estimf, self._input_target)
            self._loss_gti = self._criterion(estimi, self._input_target)
            self._compute_metric(estimf)
            total_loss = self._loss_gtf+self._loss_gti
        else:
            total_loss = -1

        self._compute_movement(estimf)

        # keep visuals
        if keep_data_for_visuals:
            self._keep_data(estimf)

        return total_loss

    def _estimate(self, img):
        return self._reg.to(self._device_master).forward(img)

    def _keep_data(self, estim):
        predicted = estim.max(1)[1].detach().cpu().numpy()
        vis_img = util.tensor2im(self._input_img.detach(), unnormalize=True, to_numpy=True)
        self._vis_input_img = plot_estim(vis_img, predicted, self._input_target.detach().cpu().numpy())

    def get_image_paths(self):
        return OrderedDict()

    def get_current_errors(self):
        loss_dict = OrderedDict()
        loss_dict["loss_gt"] = self._loss_gtf.item()
        loss_dict["unloss_gt"] = self._metric.item()
        loss_dict["unloss_euler_gt"] = self._metricEuler.item()
        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr', self._current_lr)])

    def get_current_visuals(self):
        visuals = OrderedDict()
        visuals["1_estim_img"] = self._vis_input_img
        return visuals

    def get_current_moves(self):
        moves_dict = OrderedDict()
        moves_dict["moves_gt"] = self._inputUn
        moves_dict["moves_predicted"]=self._estimUn
        return moves_dict

    def get_current_betas(self):
        betas = OrderedDict()
        betas["betas"] = self._betas
        return betas

    def save(self, epoch_label, save_type, do_remove_prev=True):
        # save networks
        self._save_network(self._reg, 'nn_reg', epoch_label, save_type, do_remove_prev)
        self._save_optimizer(self._optimizer, 'o_reg', epoch_label, save_type, do_remove_prev)

    def load(self):
        # load networks
        load_epoch = self._opt["model"]["load_epoch"]
        self._load_network(self._reg, 'nn_reg', load_epoch)
        if self._is_train:
            self._load_optimizer(self._optimizer, "o_reg", load_epoch)

    def update_learning_rate(self, curr_epoch):
        initial_lr = float(self._opt["train"]["reg_lr"])
        nepochs_no_decay = self._opt["train"]["nepochs_no_decay"]
        nepochs_decay = self._opt["train"]["nepochs_decay"]
        # update lr
        if curr_epoch <= nepochs_no_decay:
            self._current_lr = initial_lr
        else:
            new_lr = self._lr_linear(self._current_lr, nepochs_decay, initial_lr)
            self._update_learning_rate(self._optimizer, "reg", self._current_lr, new_lr)
            self._current_lr = new_lr

    def _compute_metric(self,estim):
        #unormalize
        if self._Id == 96:
            estimUn = (estim.view(self._B,self._Sd,self._Idr,self._Idc)*self._std)+self._mean
            inputUn = (self._input_target.view(self._B,self._Sd,self._Idr,self._Idc)*self._std)+self._mean
        else:
            estimUn = (estim.view(self._B,self._Sd,self._Idr*self._Idc)*self._std)+self._mean
            inputUn = (self._input_target.view(self._B,self._Sd,self._Idr*self._Idc)*self._std)+self._mean

        estimUnEuler = pytorchangles.rotmat_to_euler(pytorchangles.expmap_to_rotmat(estimUn, self._device_master),
                                            self._device_master)

        inputUnEuler = pytorchangles.rotmat_to_euler(pytorchangles.expmap_to_rotmat(inputUn, self._device_master),
                                            self._device_master)

        if self._loss_type == "euclidean":
            #Euclidean Distance
            self._metric = torch.mean(torch.sqrt(torch.sum((inputUn-estimUn)**2,dim=-1)))
            self._metricEuler = torch.mean(torch.sqrt(torch.sum((inputUnEuler-estimUnEuler)**2,dim=-1)))
        else:#DEFAULT
            #MSE Square Error
            self._metric = self._criterion(estimUn, inputUn)
            self._metricEuler = self._criterion(estimUnEuler, inputUnEuler)

    def _compute_movement(self,estim):
        #unormalize
        if self._Id == 96:
            estimUn = (estim.view(self._B,self._Sd,self._Idr,self._Idc)*self._std)+self._mean
            inputUn = (self._global_translation_target.view(self._B,self._Sd,self._Idr,self._Idc)*self._std)+self._mean
        else:
            estimUn = (estim.view(self._B,self._Sd,self._Idr*self._Idc)*self._std)+self._mean
            inputUn = (self._global_translation_target.view(self._B,self._Sd,self._Idr*self._Idc)*self._std)+self._mean

        #Moves for Visualization
        self._estimUn = estimUn
        self._inputUn = inputUn

        if self._inputType == "hidden50FramesInput69par" or self._inputType == "repeatedFrame50Input69par":
            self._estimUn[:,:,0:3] = self._global_translation_img[:,:,0:3]*self._std[0:3]+self._mean[0:3]
        else:
            self._estimUn[:,:,0:6] = self._global_translation_img[:,:,0:6]*self._std[0:6]+self._mean[0:6]

        self._betas = self._betas1
