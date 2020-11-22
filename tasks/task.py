import logging

import torch
from torch import optim, nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import transforms

from models.model import Model
from tasks.batch import Batch
from utils.parameters import Params

logger = logging.getLogger('logger')


class Task:
    params: Params = None

    train_dataset = None
    test_dataset = None
    train_loader = None
    test_loader = None
    classes = None

    model: Model = None
    optimizer: optim.Optimizer = None
    criterion: nn.Module = None
    scheduler: MultiStepLR = None

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    "Generic normalization for input data."
    input_shape: torch.Size = None

    def __init__(self, params: Params):
        self.params = params
        self.init_task()

    def init_task(self):
        self.load_data()
        self.build_model()
        self.resume_model()
        self.model = self.model.to(self.params.device)

        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()
        self.set_input_shape()

    def load_data(self) -> None:
        raise NotImplemented

    def build_model(self) -> None:
        raise NotImplemented

    def get_criterion(self) -> Module:
        """Initialize with Cross Entropy by default.

        We use reduction `none` to support gradient shaping defense.
        :return:
        """
        return nn.CrossEntropyLoss(reduction='none')

    def get_optimizer(self, model=None) -> Optimizer:
        if self.params.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(),
                                      lr=self.params.lr,
                                      weight_decay=self.params.decay,
                                      momentum=self.params.momentum)
        elif self.params.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(),
                                       lr=self.params.lr,
                                       weight_decay=self.params.decay)
        else:
            raise ValueError(f'No optimizer: {self.optimizer}')

        return optimizer

    def make_scheduler(self) -> None:
        if self.params.scheduler:
            self.scheduler = MultiStepLR(self.optimizer,
                                         milestones=self.params.scheduler_milestones,
                                         last_epoch=self.params.start_epoch,
                                         gamma=0.1)

    def resume_model(self):
        if self.params.resume_model:
            logger.info(f'Resuming training from {self.params.resume_model}')
            loaded_params = torch.load(f"saved_models/"
                                       f"{self.params.resume_model}")
            self.model.load_state_dict(loaded_params['state_dict'])
            self.params.start_epoch = loaded_params['epoch']
            self.params.lr = loaded_params.get('lr', self.params.lr)

            logger.warning(f"Loaded parameters from saved model: LR is"
                           f" {self.params.lr} and current epoch is"
                           f" {self.params.start_epoch}")

    def set_input_shape(self):
        inp = self.train_dataset[0][0]
        self.params.input_shape = inp.shape

    def get_batch(self, batch_id, data) -> Batch:
        """Process data into a batch.

        Specific for different datasets and data loaders this method unifies
        the output by returning the object of class Batch.
        :param batch_id: id of the batch
        :param data: object returned by the Loader.
        :return:
        """
        inputs, labels = data
        batch = Batch(batch_id, inputs, labels)
        return batch.to(self.params.device)

    @staticmethod
    def get_batch_accuracy(outputs, labels, top_k=(1,)):
        """Computes the precision@k for the specified values of k"""
        max_k = max(top_k)
        batch_size = labels.size(0)

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        if len(res) == 1:
            res = res[0]
        return res
