import math
import random
from copy import deepcopy
from typing import List, Any, Dict

from metrics.accuracy_metric import AccuracyMetric
from metrics.test_loss_metric import TestLossMetric
from tasks.fl.fl_user import FLUser
import torch
import logging
from torch.nn import Module

from tasks.task import Task
logger = logging.getLogger('logger')


class FederatedLearningTask(Task):
    fl_train_loaders: List[Any] = None
    ignored_weights = ['num_batches_tracked']#['tracked', 'running']
    adversaries: List[int] = None

    def init_task(self):
        self.load_data()
        self.model = self.build_model()
        self.resume_model()
        self.model = self.model.to(self.params.device)

        self.local_model = self.build_model().to(self.params.device)
        self.criterion = self.make_criterion()
        self.adversaries = self.sample_adversaries()

        self.metrics = [AccuracyMetric(), TestLossMetric(self.criterion)]
        self.set_input_shape()
        return

    def get_empty_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator

    def sample_users_for_round(self, epoch) -> List[FLUser]:
        sampled_ids = random.sample(
            range(self.params.fl_total_participants),
            self.params.fl_no_models)
        sampled_users = []
        for pos, user_id in enumerate(sampled_ids):
            train_loader = self.fl_train_loaders[user_id]
            compromised = self.check_user_compromised(epoch, pos, user_id)
            user = FLUser(user_id, compromised=compromised,
                          train_loader=train_loader)
            sampled_users.append(user)

        return sampled_users

    def check_user_compromised(self, epoch, pos, user_id):
        """Check if the sampled user is compromised for the attack.

        If single_epoch_attack is defined (eg not None) then ignore
        :param epoch:
        :param pos:
        :param user_id:
        :return:
        """
        compromised = False
        if self.params.fl_single_epoch_attack is not None:
            if epoch == self.params.fl_single_epoch_attack:
                if pos < self.params.fl_number_of_adversaries:
                    compromised = True
                    logger.warning(f'Attacking once at epoch {epoch}. Compromised'
                                   f' user: {user_id}.')
        else:
            compromised = user_id in self.adversaries
        return compromised

    def sample_adversaries(self) -> List[int]:
        adversaries_ids = []
        if self.params.fl_number_of_adversaries == 0:
            logger.warning(f'Running vanilla FL, no attack.')
        elif self.params.fl_single_epoch_attack is None:
            adversaries_ids = random.sample(
                range(self.params.fl_total_participants),
                self.params.fl_number_of_adversaries)
            logger.warning(f'Attacking over multiple epochs with following '
                           f'users compromised: {adversaries_ids}.')
        else:
            logger.warning(f'Attack only on epoch: '
                           f'{self.params.fl_single_epoch_attack} with '
                           f'{self.params.fl_number_of_adversaries} compromised'
                           f' users.')

        return adversaries_ids
    def get_model_optimizer(self, model):
        local_model = deepcopy(model)
        local_model = local_model.to(self.params.device)

        optimizer = self.make_optimizer(local_model)

        return local_model, optimizer

    def copy_params(self, global_model, local_model):
        local_state = local_model.state_dict()
        for name, param in global_model.state_dict().items():
            if name in local_state and name not in self.ignored_weights:
                local_state[name].copy_(param)

    def get_fl_update(self, local_model, global_model) -> Dict[str, torch.Tensor]:
        local_update = dict()
        for name, data in local_model.state_dict().items():
            if self.check_ignored_weights(name):
                continue
            local_update[name] = (data - global_model.state_dict()[name])

        return local_update

    def accumulate_weights(self, weight_accumulator, local_update):
        update_norm = self.get_update_norm(local_update)
        for name, value in local_update.items():
            self.dp_clip(value, update_norm)
            weight_accumulator[name].add_(value)

    def update_global_model(self, weight_accumulator, global_model: Module):
        for name, sum_update in weight_accumulator.items():
            if self.check_ignored_weights(name):
                continue
            scale = self.params.fl_eta / self.params.fl_total_participants
            average_update = scale * sum_update
            self.dp_add_noise(average_update)
            model_weight = global_model.state_dict()[name]
            model_weight.add_(average_update)

    def dp_clip(self, local_update_tensor: torch.Tensor, update_norm):
        if self.params.fl_diff_privacy and \
                update_norm > self.params.fl_dp_clip:
            norm_scale = self.params.fl_dp_clip / update_norm
            local_update_tensor.mul_(norm_scale)

    def dp_add_noise(self, sum_update_tensor: torch.Tensor):
        if self.params.fl_diff_privacy:
            noised_layer = torch.FloatTensor(sum_update_tensor.shape)
            noised_layer = noised_layer.to(self.params.device)
            noised_layer.normal_(mean=0, std=self.params.fl_dp_noise)
            sum_update_tensor.add_(noised_layer)

    def get_update_norm(self, local_update):
        squared_sum = 0
        for name, value in local_update.items():
            if self.check_ignored_weights(name):
                continue
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        update_norm = math.sqrt(squared_sum)
        return update_norm

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False
