from tasks.batch import Batch
from tasks.task import Task
from utils.parameters import Params


class Synthesizer:
    params: Params
    task: Task

    def __init__(self, task: Task):
        self.task = task
        self.params = task.params

    def make_backdoor_batch(self, batch: Batch, test=False, attack=True) -> Batch:

        # Don't attack if only normal loss task.
        if (not attack) or (self.params.loss_tasks == ['normal'] and not test):
            return batch

        if test:
            attack_portion = batch.batch_size
        else:
            attack_portion = round(
                batch.batch_size * self.params.poisoning_proportion)

        backdoored_batch = batch.clone()
        sub_batch = backdoored_batch.clip(attack_portion)
        self.apply_backdoor(sub_batch)

        return backdoored_batch

    def apply_backdoor(self, batch):
        """
        Modifies only a portion of the batch (represents batch poisoning).

        :param batch:
        :return:
        """
        self.synthesize_inputs(batch=batch)
        self.synthesize_labels(batch=batch)

        return

    def synthesize_inputs(self, batch):
        raise NotImplemented

    def synthesize_labels(self, batch):
        raise NotImplemented
