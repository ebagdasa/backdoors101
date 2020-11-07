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

    def attack_batch(self, batch: Batch) -> Batch:
        attack_portion = self.params.poisoning_proportion * batch.batch_size

        return self.apply_backdoor(batch.clone(), round(attack_portion))

    def apply_backdoor(self, batch, attack_portion):
        raise NotImplemented