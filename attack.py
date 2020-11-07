import logging

from backdoors.backdoor import Backdoor
from losses.loss_functions import compute_all_losses_and_grads
from utils.min_norm_solvers import MGDASolver
from utils.parameters import Params

logger = logging.getLogger('logger')


class Attack:
    params: Params
    backdoor: Backdoor

    def __init__(self, params, backdoor):
        self.params = params
        self.backdoor = backdoor

        # NC hyper params
        self.mask = None
        self.pattern = None

        # Model before attack
        self.fixed_model = None

    def compute_blind_loss(self, model, criterion, batch):
        batch = batch.clip(self.params.clip_batch)
        loss_tasks = self.params.loss_tasks.copy()
        batch_back = self.backdoor.attack_batch(batch)
        scale = dict()

        if len(loss_tasks) == 1:
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, compute_grad=False
            )
        elif self.params.loss_balance == 'MGDA':

            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, compute_grad=True)

            if len(loss_tasks) > 1:
                scale = MGDASolver.get_scales(grads, loss_values,
                                              self.params.mgda_normalize,
                                              loss_tasks)
        elif self.params.loss_balance == 'fixed':
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, compute_grad=False)

            for t in loss_tasks:
                scale[t] = self.params.fixed_scales[t]
        else:
            raise ValueError(f'Please choose between `MGDA` and `fixed`.')

        if len(loss_tasks) == 1:
            scale = {loss_tasks[0]: 1.0}

        blind_loss = self.scale_losses(loss_tasks, loss_values, scale)

        return blind_loss

    def scale_losses(self, loss_tasks, loss_values, scale):
        blind_loss = 0
        for it, t in enumerate(loss_tasks):
            self.params.running_losses[t].append(loss_values[t].item())
            self.params.running_scales[t].append(scale[t])
            if it == 0:
                blind_loss = scale[t] * loss_values[t]
            else:
                blind_loss += scale[t] * loss_values[t]
        self.params.running_losses['total'].append(blind_loss.item())
        return blind_loss
