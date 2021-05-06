import torch.nn as nn
import torch
from torch.nn import Parameter
from utils.utils import th, thp
from models.model import Model


class NCModel(Model):

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.pattern = torch.zeros([self.size , self.size ], requires_grad=True)\
                             + torch.normal(0, 0.5, [self.size , self.size ])
        self.mask = torch.zeros([self.size , self.size ], requires_grad=True)
                   # + torch.normal(0, 2, [self.size , self.size ])
        self.mask = Parameter(self.mask)
        self.pattern = Parameter(self.pattern)

    def forward(self, x, latent=None):
        maskh = th(self.mask)
        patternh = thp(self.pattern)
        x = (1 - maskh) * x + maskh * patternh

        return x

    def re_init(self, device):
        p = torch.zeros([self.size , self.size ], requires_grad=True)\
                             + torch.normal(0, 0.5, [self.size , self.size ])

        self.pattern.data = p.to(device)
        m = torch.zeros([self.size , self.size ], requires_grad=True)

        self.mask.data = m.to(device)