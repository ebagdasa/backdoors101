import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax
from torchvision import transforms

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


logger = logging.getLogger("logger")

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])

class ImageHelper(Helper):
    classes = None
    train_loader = None
    test_loader = None


    def load_multimnist(self, batch_size):
        self.train_dataset = MNIST(root='./data', train=True, download=True, transform=global_transformer(),
                          multi=True)
        self.train_loader = torch_data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=4)

        self.test_dataset = MNIST(root='./data', train=False, download=True, transform=global_transformer(),
                        multi=True)
        self.test_loader = torch_data.DataLoader(self.test_dataset, batch_size=100, shuffle=True, num_workers=4)
        self.classes = list(range(100))


    def load_cifar10(self, batch_size):

        if self.transform_train:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if self.smoothing:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        if self.poison_images:
            self.train_loader = self.poison_loader()
        else:
            self.train_loader = torch_data.DataLoader(self.train_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                                                 shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return True


    def load_pipa(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


        self.train_dataset = PipaDataset(train=True, transform=train_transform)

        poison_weights = [0.99, 0.004, 0.004, 0.004, 0.004]

        weights = [14081, 4893, 1779, 809, 862] #[0.62, 0.22, 0.08, 0.04, 0.04]
        weights = torch.tensor([0.03, 0.07, 0.2,0.35,0.35])
        weights_labels = weights[self.train_dataset.labels]
        sampler = torch_data.sampler.WeightedRandomSampler(weights_labels, len(self.train_dataset))
        self.train_loader = torch_data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                              sampler=sampler)


        # self.train_loader = torch_data.DataLoader(self.train_dataset, batch_size=self.batch_size,
        #                                           shuffle=True, num_workers=2)
        self.test_dataset = PipaDataset(train=False, transform=test_transform)
        self.test_loader = torch_data.DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                                                 shuffle=False, num_workers=2)

        self.classes = list(range(5))

    def load_imagenet(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        self.train_dataset = torchvision.datasets.ImageNet(root='/media/ssd/eugene/datasets/imagenet/', download=False,
                                                split='train', transform=train_transform)

        self.test_dataset = torchvision.datasets.ImageNet(root='/media/ssd/eugene/datasets/imagenet/', download=False,
                                                           split='val',  transform=test_transform)

        self.train_loader = torch_data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=2)
        self.test_loader = torch_data.DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                                                       shuffle=False, num_workers=2)

        with open('/media/ssd/eugene/datasets/imagenet/imagenet1000_clsidx_to_labels.txt') as f:
            self.classes = eval(f.read())


    def poison_loader(self):

        all_images = set(range(len(self.train_dataset)))
        unpoisoned_images = list(all_images.difference(set(self.poison_images)))

        return torch_data.DataLoader(self.train_dataset,
                                    batch_size=self.batch_size,
                                    sampler=torch_data.sampler.SubsetRandomSampler(unpoisoned_images))

    def load_mnist(self, batch_size):
        transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

        transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

        self.train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                          download=True, transform=transform_train)
        self.train_loader = torch_data.DataLoader(self.train_dataset, batch_size=batch_size,
                                                        shuffle=True, num_workers=2)
        self.test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                         download=True, transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset, batch_size=100,
                                                       shuffle=False, num_workers=2)

        self.classes = (0,1,2,3,4,5,6,7,8,9)

        return True


    def load_text(self):
        from transformers import BertTokenizer
        from torchtext import data, datasets
        import dill


        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        init_token = tokenizer.cls_token
        eos_token = tokenizer.sep_token
        pad_token = tokenizer.pad_token
        unk_token = tokenizer.unk_token
        init_token_idx = tokenizer.cls_token_id
        eos_token_idx = tokenizer.sep_token_id
        pad_token_idx = tokenizer.pad_token_id
        unk_token_idx = tokenizer.unk_token_id

        def tokenize_and_cut(sentence):
            tokens = tokenizer.tokenize(sentence)
            tokens = tokens[:max_input_length - 2]
            return tokens


        TEXT = data.Field(batch_first=True,
                          use_vocab=False,
                          tokenize=tokenize_and_cut,
                          preprocessing=tokenizer.convert_tokens_to_ids,
                          init_token=init_token_idx,
                          eos_token=eos_token_idx,
                          pad_token=pad_token_idx,
                          unk_token=unk_token_idx)

        LABEL = data.LabelField(dtype=torch.float)

        max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']


        self.train_data = datasets.imdb.IMDB('.data', TEXT, LABEL)
        self.test_data = datasets.imdb.IMDB('.data', TEXT, LABEL)
        with open('./data/nlp/train_data.dill', 'rb') as f:
            self.train_data.examples = dill.load(f)
        with open('./data/nlp/test_data.dill', 'rb') as f:
            self.test_data.examples = dill.load(f)
        random.seed(5)
        # self.train_data.examples = random.sample(self.train_data.examples, 20000)
        self.test_data.examples = random.sample(self.test_data.examples, 5000)

        LABEL.build_vocab(self.train_data)
        # LABEL.build_vocab(self.test_data)

        self.train_loader, self.test_loader = data.BucketIterator.splits(
            (self.train_data, self.test_data),
            batch_size=self.batch_size,
            device=self.device)

