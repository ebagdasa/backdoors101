import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

from models.resnet import resnet18
from tasks.task import Task


class Cifar10Task(Task):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    def load_data(self):
        self.load_cifar_data()

    def load_cifar_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)
        if self.params.poison_images:
            self.train_loader = self.remove_semantic_backdoors()
        else:
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           num_workers=0)
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return True

    def build_model(self) -> nn.Module:
        if self.params.pretrained:
            model = resnet18(pretrained=True)

            # model is pretrained on ImageNet changing classes to CIFAR
            model.fc = nn.Linear(512, len(self.classes))
        else:
            model = resnet18(pretrained=False,
                                  num_classes=len(self.classes))
        return model

    def remove_semantic_backdoors(self):
        """
        Semantic backdoors still occur with unmodified labels in the training
        set. This method removes them, so the only occurrence of the semantic
        backdoor will be in the
        :return: None
        """

        all_images = set(range(len(self.train_dataset)))
        unpoisoned_images = list(all_images.difference(set(
            self.params.poison_images)))

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       sampler=SubsetRandomSampler(
                                           unpoisoned_images))
