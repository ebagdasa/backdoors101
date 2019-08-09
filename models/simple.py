import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import datetime
import random


def reseed(seed=5):
    seed = 5
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


class SimpleNet(nn.Module):
    def __init__(self, name=None, created_time=None):
        super(SimpleNet, self).__init__()
        self.created_time = created_time
        self.name=name
        reseed()



    def visualize(self, vis, epoch, acc, loss=None, eid='main', is_dp=False, name=None):
        if name is None:
            name = self.name + '_poisoned' if is_dp else self.name
        vis.line(X=np.array([epoch]), Y=np.array([acc]), name=name, win='vacc_{0}'.format(self.created_time), env=eid,
                                update='append' if vis.win_exists('vacc_{0}'.format(self.created_time), env=eid) else None,
                                opts=dict(showlegend=True, title='Accuracy_{0}'.format(self.created_time),
                                          width=700, height=400))
        if loss is not None:
            vis.line(X=np.array([epoch]), Y=np.array([loss]), name=name, env=eid,
                                     win='vloss_{0}'.format(self.created_time),
                                     update='append' if vis.win_exists('vloss_{0}'.format(self.created_time), env=eid) else None,
                                     opts=dict(showlegend=True, title='Loss_{0}'.format(self.created_time), width=700, height=400))

        return



    def train_vis(self, vis, epoch, data_len, batch, loss, eid='main', name=None, win='vtrain'):

        vis.line(X=np.array([(epoch-1)*data_len+batch]), Y=np.array([loss]),
                                 env=eid,
                                 name=f'{name}' if name is not None else self.name, win=f'{win}_{self.created_time}',
                                 update='append' if vis.win_exists(f'{win}_{self.created_time}', env=eid) else None,
                                 opts=dict(showlegend=True, width=700, height=400, title='Train loss_{0}'.format(self.created_time)))



    def save_stats(self, epoch, loss, acc):
        self.stats['epoch'].append(epoch)
        self.stats['loss'].append(loss)
        self.stats['acc'].append(acc)


    def copy_params(self, state_dict, coefficient_transfer=100):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                #
                random_tensor = (torch.cuda.FloatTensor(shape).random_(0, 100) <= coefficient_transfer).type(
                    torch.cuda.FloatTensor)
                negative_tensor = (random_tensor*-1)+1
                # own_state[name].copy_(param)
                own_state[name].copy_(param.clone())




class Net(SimpleNet):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FlexiNet(SimpleNet):
    def __init__(self, input_channel, output_dim):
        super(FlexiNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(13 * 13 * 50, 500)
        self.fc2 = nn.Linear(500, output_dim)

    def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 13 * 13 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

#
  # input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
  # y = tf.keras.layers.Conv2D(16, 8,
  #                            strides=2,
  #                            padding='same',
  #                            activation='relu').apply(input_layer)
  # y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  # y = tf.keras.layers.Conv2D(32, 4,
  #                            strides=2,
  #                            padding='valid',
  #                            activation='relu').apply(y)
  # y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  # y = tf.keras.layers.Flatten().apply(y)
  # y = tf.keras.layers.Dense(32, activation='relu').apply(y)
  # logits = tf.keras.layers.Dense(10).apply(y)