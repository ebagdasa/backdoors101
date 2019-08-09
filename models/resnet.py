import torch.nn as nn
import torchvision.models as models

from models.simple import SimpleNet


class Res(SimpleNet):
    def __init__(self, cifar10=True):
        super(Res, self).__init__()
        if cifar10:
            self.res = models.resnet18(num_classes=10)
        else:
            self.res = models.resnet18(num_classes=100)


    def forward(self, x):
        x = self.res(x)
        return x


class PretrainedRes(SimpleNet):
    def __init__(self, no_classes):
        super(PretrainedRes, self).__init__()
        self.res = models.resnet101(pretrained=True)
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, no_classes)


    def forward(self, x):
        x = self.res(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x