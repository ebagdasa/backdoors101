import logging
import time
from collections import defaultdict
import yaml
from models.simple import Discriminator
from utils.data_helper import DataHelper
from utils.msssim import MSSSIM
from utils.parameters import Params
from utils.utils import th, create_logger

logger = logging.getLogger('logger')
from torch.nn.functional import log_softmax
from shutil import copyfile

import math
import torch
import random
import numpy as np
import os
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd


class Helper:
    params: Params

    def __init__(self, params):
        self.params = Params(**params)
        self.data = DataHelper(self.params)


        self.times = {'backward': list(), 'forward': list(), 'step': list(),
                      'scales': list(), 'total': list(), 'poison': list()}
        self.save_dict = defaultdict(list)

        self.make_folders()
        self.nc = True if 'nc_adv' in self.params.get('losses', list()) else False
        self.mixed = None
        self.mixed_optim = None

        self.nc_tensor_weight = torch.zeros(1000).cuda()
        self.nc_tensor_weight[self.params.backdoor_label] = 1.0


    def make_folders(self):
        logger = create_logger()
        if self.log:
            try:
                os.mkdir(self.folder_path)
            except FileExistsError:
                logger.info('Folder already exists')

            with open('saved_models/runs.html', 'a') as f:
                f.writelines([f'<div><a href="https://github.com/ebagdasa/backdoors/tree/{self.params.commit}">GitHub</a>,'
                              f'<span> <a href="http://gpu/'
                              f'{self.params.folder_path}">{self.params.name}_{self.params.current_time}</a></div>'])

            fh = logging.FileHandler(filename=f'{self.params.folder_path}/log.txt')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

            logger.warning(f'Logging things. current path: {self.params.folder_path}')
            logger.error(
                f'LINK: <a href="https://github.com/ebagdasa/backdoors/tree/{self.params.commit}">https://github.com/ebagdasa/backdoors/tree/{helper.commit}</a>')

            with open(f'{self.params.folder_path}/params.yaml.txt', 'w') as f:
                yaml.dump(self.params, f)

        if self.params.tb:
            wr = SummaryWriter(log_dir=f'runs/{self.params.name}')
            self.writer = wr
            table = create_table(self.params.to_dict())
            helper.writer.add_text('Model Params', table)



    def save_model(self, model=None, epoch=0, val_loss=0):

        if self.params['save_model'] and self.log:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params.get('save_on_epochs', []):
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def save_mixed(self, epoch):
        model_name = '{0}/model_mixed.pt.tar'.format(self.params['folder_path'])
        saved_dict = {'state_dict': self.mixed.state_dict(), 'epoch': epoch,
                      'lr': self.params['lr']}
        self.save_checkpoint(saved_dict, False, model_name)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    def record_time(self, t=None, name=None):
        if t and name and self.timing == name or self.timing == True:
            torch.cuda.synchronize()
            self.times[name].append(round(1000*(time.perf_counter()-t)))


    def get_optimizer(self, model):
        if self.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=self.lr,
                                  weight_decay=self.decay, momentum=self.momentum)
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.decay)
        else:
            raise ValueError(f'No optimizer: {self.optimizer}')

        return optimizer

    def check_resume_training(self, model, lr=False):
        from models.resnet import ResNet18

        if self.resume_model:
            logger.info('Resuming training...')
            loaded_params = torch.load(f"saved_models/{self.resume_model}")
            model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']
            if lr:
                self.lr = loaded_params.get('lr', self.lr)
                print('current lr')

            # self.fixed_model = ResNet18()
            self.fixed_model.load_state_dict(loaded_params['state_dict'])
            self.fixed_model.to(self.device)

            logger.warning(f"Loaded parameters from saved model: LR is"
                        f" {self.lr} and current epoch is {self.start_epoch}")

    def flush_writer(self):
        if self.writer:
            self.writer.flush()

    def plot(self, x, y, name):
        if self.writer is not None:
            self.writer.add_scalar(tag=name, scalar_value=y, global_step=x)
            self.flush_writer()
        else:
            return False

    @staticmethod
    def fix_random(seed=1):
        # logger.warning('Setting random_seed seed for reproducible results.')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        return True

    @staticmethod
    def copy_grad(model: nn.Module):
        grads = list()
        for name, params in model.named_parameters():
            if params.requires_grad:
                grads.append(params.grad.clone().detach())
        model.zero_grad()
        return grads
