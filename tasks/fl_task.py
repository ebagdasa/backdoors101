import math
import random
from copy import deepcopy
from typing import List, Any, Dict

import torch
from torch.nn import Module

from tasks.task import Task


class FederatedLearningTask(Task):
    fl_train_loaders: List[Any] = None
    fl_test_loaders: List[Any] = None
    ignored_weights = ['tracked', 'running']

    def init_task(self):
        self.load_data()
        self.build_model()
        self.resume_model()

        self.model = self.model.to('cpu')

        self.set_input_shape()
        return

    def get_empty_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
        return

    def sample_users_for_round(self):
        user_ids = random.sample(
            range(self.params.fl_total_participants),
            self.params.fl_no_models)

        datasets = [(x, self.fl_test_loaders[x]) for x in user_ids]

        return datasets

    def get_local_model_optimizer(self):
        local_model = deepcopy(self.model)
        local_model = local_model.to(self.params.device)

        optimizer = self.get_optimizer(local_model)

        return local_model, optimizer

    def get_fl_update(self, local_model, global_model) -> Dict[str, torch.Tensor]:
        local_update = dict()
        for name, data in local_model.state_dict().items():
            if self.check_ignored_weights(name):
                continue
            local_update[name].add_(data - global_model.state_dict()[name])

        return local_update

    def accumulate_weights(self, weight_accumulator, local_update):
        update_norm = self.get_update_norm(local_update)
        for name, value in local_update.items():
            self.dp_clip(value, update_norm)
            weight_accumulator[name].add_(value)

    def update_global_model(self, weight_accumulator, global_model: Module):
        for name, sum_update in weight_accumulator.items():
            scale = self.params.fl_eta / self.params.fl_total_participants
            average_update = scale * sum_update
            self.dp_add_noise(average_update)
            global_model.state_dict()[name].add_(average_update)


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
