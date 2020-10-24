from utils.helper import Helper
import logging
from data_helpers.datasets.pipa import *
# from models.word_model import RNNModel
# from utils.nlp_dataset import NLPDataset
# from utils.text_load import *


logger = logging.getLogger("logger")

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])

class ImageHelper(Helper):
    classes = None
    train_loader = None
    test_loader = None




