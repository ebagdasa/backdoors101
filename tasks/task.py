import logging
import torch.utils.data as torch_data
from torch import optim, nn
from torch.nn import CrossEntropyLoss

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

    def __init__(self, params: Params):
        self.params = params
        self.load_data()
        self.build_model()
        self.make_optimizer()
        self.make_criterion()

    def load_data(self) -> None:
        raise NotImplemented

    def build_model(self) -> None:
        raise NotImplemented

    def make_criterion(self) -> None:
        """Initialize with Cross Entropy by default.

        We use reduction `none` to support gradient shaping defense.
        :return:
        """
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def make_optimizer(self) -> None:
        if self.params.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                      lr=self.params.lr,
                                  weight_decay=self.params.decay,
                                  momentum=self.params.momentum)
        elif self.params.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.params.lr,
                                   weight_decay=self.params.decay)
        else:
            raise ValueError(f'No optimizer: {self.optimizer}')

    def get_batch(self, data) -> Batch:
        """Process data into a batch.

        Specific for different datasets and data loaders this method unifies
        the output by returning the object of class Batch.
        :param data: object returned by the Loader.
        :return:
        """
        inputs, labels = data
        batch = Batch(inputs, labels)
        return batch.to(self.params.device)

