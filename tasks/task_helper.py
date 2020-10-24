from data_helpers.batch import Batch
from data_helpers.datasets.celeba import CelebA
from data_helpers.datasets.multi_mnist_loader import MNIST
from data_helpers.datasets.vggface import VGG_Faces2
from utils.parameters import Params
from data_helpers.datasets.pipa import *

import logging
import torch.utils.data as torch_data
import torchvision

logger = logging.getLogger('logger')



class TaskHelper:
    params: Params
    train_dataset = None
    test_dataset = None
    train_loader = None
    test_loader = None
    classes = None
    model = None

    def __init__(self, params: Params):
        self.params = params

    def load_data(self):
        return NotImplemented

    def iter_batches(self, train=True):
        data_loader = self.train_loader if train else self.test_loader
        for input, label in data_loader:
            yield Batch(input, label)


    def remove_semantic_backdoors(self):
        """
        Semantic backdoors still occur with unmodified labels in the training
        set. This method removes them, so the only occurrence of the semantic
        backdoor will be in the
        :return:
        """

        all_images = set(range(len(self.train_dataset)))
        unpoisoned_images = list(all_images.difference(set(
            self.params.poison_images)))

        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                     batch_size=self.params.batch_size,
                                     sampler=torch_data.sampler.SubsetRandomSampler(
                                         unpoisoned_images))

