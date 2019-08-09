import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax
from torchvision import transforms

from utils.helper import Helper
import random
import logging
import torchvision
# from models.word_model import RNNModel
# from utils.nlp_dataset import NLPDataset
# from utils.text_load import *

logger = logging.getLogger("logger")


class ImageHelper(Helper):
    classes = None
    train_loader = None
    test_loader = None

    def load_cifar10(self, batch_size):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=10,
                                                 shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return True
