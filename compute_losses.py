from typing import List

from utils.helper import Helper
import torch.nn as nn
import torch

def compute_losses(helper: Helper, model: nn.Module, data: List[torch.Tensor]):

    if 'backdoor' in helper.params.loss_tasks:
        data_backdoor = helper.backdoor_fun(data)

    input, labels = data[0], data[1]

