import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax
from torchvision import transforms

from data.vggface import VGG_Faces2
from utils.helper import Helper
import random
import logging
import torchvision
from utils.pipa_loader import *
# from models.word_model import RNNModel
# from utils.nlp_dataset import NLPDataset
# from utils.text_load import *
import torch.utils.data as torch_data
from data.multi_mnist_loader import MNIST
from data.celeba import CelebA


logger = logging.getLogger("logger")

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])

class ImageHelper(Helper):
    classes = None
    train_loader = None
    test_loader = None




