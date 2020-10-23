from data.celeba import CelebA
from data.multi_mnist_loader import MNIST
from data.vggface import VGG_Faces2
from utils.parameters import Params
from utils.pipa_loader import *

import logging
import torch.utils.data as torch_data
import torchvision

logger = logging.getLogger('logger')



class DataHelper:
    params: Params
    train_dataset = None
    test_dataset = None
    train_loader = None
    test_loader = None
    classes = None

    def __init__(self, params: Params):
        self.params = params
        if self.params.dataset == 'mnist':
            self.load_mnist()
        elif self.params.dataset == 'multimnist':
            self.load_multimnist()
        elif self.params.dataset == 'cifar10':
            self.load_cifar10()
        elif self.params.dataset == 'imagenet':
            self.load_imagenet()
        elif self.params.dataset == 'pipa':
            self.load_pipa()
        elif self.params.dataset == 'nlp':
            self.load_text()
        else:
            raise ValueError(f'Provided dataset {self.params.dataset} is not '
                             f'supported yet. Choose the one from '
                             f'`data_helper.py`.')

    def load_mnist(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.train_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)
        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True, num_workers=2)
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=100,
                                                 shuffle=False, num_workers=2)
        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        return True

    def load_multimnist(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,),
                                                             (0.3081,))])
        self.train_dataset = MNIST(root='./data', train=True, download=True,
                                   transform=transform,
                                   multi=True)
        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=4)
        self.test_dataset = MNIST(root='./data', train=False, download=True,
                                  transform=transform,
                                  multi=True)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=100, shuffle=True,
                                                 num_workers=4)
        self.classes = list(range(100))

    def load_vggface(self):
        logger.error('VGG dataset is unfinished, needs more work')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.train_dataset = VGG_Faces2(
            root=self.params.data_path,
            train=True, transform=transform_train)
        self.test_dataset = VGG_Faces2(
            root=self.params.data_path,
            train=False, transform=transform_test)

        return True

    def load_cifar10(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        self.train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                          train=True,
                                                          download=True,
                                                          transform=transform_train)
        if self.params.poison_images:
            self.train_loader = self.poison_loader()
        else:
            self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                      batch_size=self.params.batch_size,
                                                      shuffle=True,
                                                      num_workers=2)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                         train=False,
                                                         download=True,
                                                         transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return True

    def load_pipa(self):
        # This is a custom task for counting people's faces on PIPA dataset.

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
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
        # poison_weights = [0.99, 0.004, 0.004, 0.004, 0.004]
        # weights = [14081, 4893, 1779, 809,
        #            862]  # [0.62, 0.22, 0.08, 0.04, 0.04]
        weights = torch.tensor([0.03, 0.07, 0.2, 0.35, 0.35])
        weights_labels = weights[self.train_dataset.labels]
        sampler = torch_data.sampler.WeightedRandomSampler(weights_labels, len(
            self.train_dataset))
        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  sampler=sampler)

        self.test_dataset = PipaDataset(train=False, transform=test_transform)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False, num_workers=2)

        self.classes = list(range(5))

    def load_imagenet(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
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

        self.train_dataset = torchvision.datasets.ImageNet(
            root=self.params.data_path,
            split='train', transform=train_transform)

        self.test_dataset = torchvision.datasets.ImageNet(
            root=self.params.data_path,
            split='val', transform=test_transform)

        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True, num_workers=2)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False, num_workers=2)

        with open(
                f'{self.params.data_path}/imagenet1000_clsidx_to_labels.txt') \
                as f:
            self.classes = eval(f.read())

    def load_celeba(self):
        logger.error('Celeba dataset is unfinished, needs more work')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            # transforms.RandomResizedCrop(178, scale=(0.9,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            # transforms.CenterCrop((178, 178)),
            # transforms.Resize((128, 128)),
            transforms.ToTensor(),
            normalize,
        ])

        self.train_dataset = CelebA(root=self.params.data_path,
                                    target_type='identity',  # ['identity',
                                    # 'attr'],
                                    split='train', transform=train_transform)

        self.test_dataset = CelebA(root=self.params.data_path,
                                   target_type='identity',
                                   split='test', transform=test_transform)

        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True, num_workers=8,
                                                  pin_memory=True)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False, num_workers=2)

    def poison_loader(self):

        all_images = set(range(len(self.train_dataset)))
        unpoisoned_images = list(all_images.difference(set(
            self.params.poison_images)))

        return torch_data.DataLoader(self.train_dataset,
                                     batch_size=self.params.batch_size,
                                     sampler=torch_data.sampler.SubsetRandomSampler(
                                         unpoisoned_images))

    def load_text(self):
        from transformers import BertTokenizer
        from torchtext import data, datasets
        import dill

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        init_token_idx = tokenizer.cls_token_id
        eos_token_idx = tokenizer.sep_token_id
        pad_token_idx = tokenizer.pad_token_id
        unk_token_idx = tokenizer.unk_token_id

        def tokenize_and_cut(sentence):
            tokens = tokenizer.tokenize(sentence)
            tokens = tokens[:max_input_length - 2]
            return tokens

        text = data.Field(batch_first=True,
                          use_vocab=False,
                          tokenize=tokenize_and_cut,
                          preprocessing=tokenizer.convert_tokens_to_ids,
                          init_token=init_token_idx,
                          eos_token=eos_token_idx,
                          pad_token=pad_token_idx,
                          unk_token=unk_token_idx)

        label = data.LabelField(dtype=torch.float)

        max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

        self.train_dataset = datasets.imdb.IMDB('.data', text, label)
        self.test_dataset = datasets.imdb.IMDB('.data', text, label)
        with open(f'{self.params.data_path}/train_data.dill', 'rb') as f:
            self.train_dataset.examples = dill.load(f)
        with open(f'{self.params.data_path}/test_data.dill', 'rb') as f:
            self.test_dataset.examples = dill.load(f)
        random.seed(5)
        self.test_dataset.examples = random.sample(self.test_dataset.examples,
                                                   5000)
        label.build_vocab(self.train_dataset)
        self.train_loader, self.test_loader = data.BucketIterator.splits(
            (self.train_dataset, self.test_dataset),
            batch_size=self.params.batch_size,
            device=self.params.device)
