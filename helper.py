import importlib
import logging
import os
import random
from collections import defaultdict
from shutil import copyfile
from typing import Union

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from attack import Attack
from synthesizers.synthesizer import Synthesizer
from tasks.fl.fl_task import FederatedLearningTask
from tasks.task import Task
from utils.parameters import Params
from utils.utils import create_logger, create_table

logger = logging.getLogger('logger')


class Helper:
    params: Params = None
    task: Union[Task, FederatedLearningTask] = None
    synthesizer: Synthesizer = None
    attack: Attack = None
    tb_writer: SummaryWriter = None

    def __init__(self, params):
        self.params = Params(**params)

        self.times = {'backward': list(), 'forward': list(), 'step': list(),
                      'scales': list(), 'total': list(), 'poison': list()}
        if self.params.random_seed is not None:
            self.fix_random(self.params.random_seed)

        self.make_folders()
        self.make_task()
        self.make_synthesizer()
        self.attack = Attack(self.params, self.synthesizer)

        self.nc = True if 'neural_cleanse' in self.params.loss_tasks else False
        self.best_loss = float('inf')

    def make_task(self):
        name_lower = self.params.task.lower()
        name_cap = self.params.task
        if self.params.fl:
            module_name = f'tasks.fl.{name_lower}_task'
            path = f'tasks/fl/{name_lower}_task.py'
        else:
            module_name = f'tasks.{name_lower}_task'
            path = f'tasks/{name_lower}_task.py'
        try:
            task_module = importlib.import_module(module_name)
            task_class = getattr(task_module, f'{name_cap}Task')
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(f'Your task: {self.params.task} should '
                                      f'be defined as a class '
                                      f'{name_cap}'
                                      f'Task in {path}')
        self.task = task_class(self.params)

    def make_synthesizer(self):
        name_lower = self.params.synthesizer.lower()
        name_cap = self.params.synthesizer
        module_name = f'synthesizers.{name_lower}_synthesizer'
        try:
            synthesizer_module = importlib.import_module(module_name)
            task_class = getattr(synthesizer_module, f'{name_cap}Synthesizer')
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(
                f'The synthesizer: {self.params.synthesizer}'
                f' should be defined as a class '
                f'{name_cap}Synthesizer in '
                f'synthesizers/{name_lower}_synthesizer.py')
        self.synthesizer = task_class(self.task)

    def make_folders(self):
        log = create_logger()
        if self.params.log:
            try:
                os.mkdir(self.params.folder_path)
            except FileExistsError:
                log.info('Folder already exists')

            with open('saved_models/runs.html', 'a') as f:
                f.writelines([f'<div><a href="https://github.com/ebagdasa/'
                              f'backdoors/tree/{self.params.commit}">GitHub'
                              f'</a>, <span> <a href="http://gpu/'
                              f'{self.params.folder_path}">{self.params.name}_'
                              f'{self.params.current_time}</a></div>'])

            fh = logging.FileHandler(
                filename=f'{self.params.folder_path}/log.txt')
            formatter = logging.Formatter('%(asctime)s - %(name)s '
                                          '- %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            log.addHandler(fh)

            log.warning(f'Logging to: {self.params.folder_path}')
            log.error(
                f'LINK: <a href="https://github.com/ebagdasa/backdoors/tree/'
                f'{self.params.commit}">https://github.com/ebagdasa/backdoors'
                f'/tree/{self.params.commit}</a>')

            with open(f'{self.params.folder_path}/params.yaml.txt', 'w') as f:
                yaml.dump(self.params, f)

        if self.params.tb:
            wr = SummaryWriter(log_dir=f'runs/{self.params.name}')
            self.tb_writer = wr
            params_dict = self.params.to_dict()
            table = create_table(params_dict)
            self.tb_writer.add_text('Model Params', table)

    def save_model(self, model=None, epoch=0, val_loss=0):

        if self.params.save_model:
            logger.info(f"Saving model to {self.params.folder_path}.")
            model_name = '{0}/model_last.pt.tar'.format(self.params.folder_path)
            saved_dict = {'state_dict': model.state_dict(),
                          'epoch': epoch,
                          'lr': self.params.lr,
                          'params_dict': self.params.to_dict()}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params.save_on_epochs:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False,
                                     filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params.save_model:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    def flush_writer(self):
        if self.tb_writer:
            self.tb_writer.flush()

    def plot(self, x, y, name):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag=name, scalar_value=y, global_step=x)
            self.flush_writer()
        else:
            return False

    def report_training_losses_scales(self, batch_id, epoch):
        if not self.params.report_train_loss or \
                batch_id % self.params.log_interval != 0:
            return
        total_batches = len(self.task.train_loader)
        losses = [f'{x}: {np.mean(y):.2f}'
                  for x, y in self.params.running_losses.items()]
        scales = [f'{x}: {np.mean(y):.2f}'
                  for x, y in self.params.running_scales.items()]
        logger.info(
            f'Epoch: {epoch:3d}. '
            f'Batch: {batch_id:5d}/{total_batches}. '
            f' Losses: {losses}.'
            f' Scales: {scales}')
        for name, values in self.params.running_losses.items():
            self.plot(epoch * total_batches + batch_id, np.mean(values),
                      f'Train/Loss_{name}')
        for name, values in self.params.running_scales.items():
            self.plot(epoch * total_batches + batch_id, np.mean(values),
                      f'Train/Scale_{name}')

        self.params.running_losses = defaultdict(list)
        self.params.running_scales = defaultdict(list)

    @staticmethod
    def fix_random(seed=1):
        from torch.backends import cudnn

        logger.warning('Setting random_seed seed for reproducible results.')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = False
        cudnn.enabled = True
        cudnn.benchmark = True
        np.random.seed(seed)

        return True
