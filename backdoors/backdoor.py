import random

from tasks.batch import Batch
from utils.parameters import Params


class Backdoor:
    params: Params
    position = (0, 0)

    def __init__(self, params):
        self.params = params

    def attack_batch(self, batch: Batch):
        if self.params.poisoning_proportion >= 1:
            self.apply_backdoor(batch)
        else:
            attack_portion = self.params.poisoning_proportion * len(batch)
            self.apply_backdoor(batch[:round(attack_portion)])

        raise NotImplemented

    def apply_backdoor(self, batch, i=-1):
        raise NotImplemented