import logging

logger = logging.getLogger('logger')

from shutil import copyfile

import math
import torch

import os


class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None
        self.dataset_size = 0
        self.train_dataset = None
        self.test_dataset = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = params
        self.name = name
        self.best_loss = math.inf
        self.folder_path = f'saved_models/model_{self.name}_{current_time}'
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')

        # if not self.params.get('environment_name', False):
        #     self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path

    def save_model(self, model=None, epoch=0, val_loss=0):
        if model is None:
            model = self.target_model
        if self.params['save_model']:
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

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def norm(parameters, max_norm):
        total_norm = 0
        for p in parameters:
            torch.sum(torch.pow(p))
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    # def compute_rdp(self):
    #     from compute_dp_sgd_privacy import apply_dp_sgd_analysis
    #
    #     N = self.dataset_size
    #     logger.info(f'Dataset size: {N}. Computing RDP guarantees.')
    #     q = self.params['batch_size'] / N  # q - the sampling ratio.
    #
    #     orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
    #               list(range(5, 64)) + [128, 256, 512])
    #
    #     steps = int(math.ceil(self.params['epochs'] * N / self.params['batch_size']))
    #
    #     apply_dp_sgd_analysis(q, self.params['z'], steps, orders, 1e-6)

    @staticmethod
    def clip_grad(parameters, max_norm, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
        return total_norm

    @staticmethod
    def clip_grad_scale_by_layer_norm(parameters, max_norm, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))

        total_norm_weight = 0
        norm_weight = dict()
        for i, p in enumerate(parameters):
            param_norm = p.data.norm(norm_type)
            norm_weight[i] = param_norm.item()
            total_norm_weight += param_norm.item() ** norm_type
        total_norm_weight = total_norm_weight ** (1. / norm_type)

        total_norm = 0
        norm_grad = dict()
        for i, p in enumerate(parameters):
            param_norm = p.grad.data.norm(norm_type)
            norm_grad[i] = param_norm.item()
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for i, p in enumerate(parameters):
                if norm_grad[i] < 1e-3:
                    continue
                scale = norm_weight[i] / total_norm_weight
                p.grad.data.mul_(math.sqrt(max_norm) * scale / norm_grad[i])
        # print(total_norm)
        # total_norm = 0
        # norm_grad = dict()
        # for i, p in enumerate(parameters):
        #     param_norm = p.grad.data.norm (norm_type)
        #     norm_grad[i] = param_norm
        #     total_norm += param_norm.item() ** norm_type
        # total_norm = total_norm ** (1. / norm_type)
        # print(total_norm)
        return total_norm



    @staticmethod
    def get_grad_vec(model, device, requires_grad=False):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if device.type == 'cpu':
            sum_var = torch.FloatTensor(size).fill_(0)
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            sum_var[size:size + layer.view(-1).shape[0]] = (layer.grad).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var