import random

from tasks.batch import Batch
from tasks.task import Task
from utils.parameters import Params


class Backdoor:
    params: Params
    task: Task

    def __init__(self, task: Task):
        self.task = task
        self.params = task.params

    def attack_batch(self, batch: Batch, test=False) -> Batch:

        # Don't attack if only normal loss task.
        if self.params.loss_tasks == ['normal']:
            return batch
        else:
            attack_portion = batch.batch_size
            if not test:
                attack_portion *= self.params.poisoning_proportion

            return self.apply_backdoor(batch.clone(), round(attack_portion))

    def apply_backdoor(self, batch, attack_portion):
        raise NotImplemented