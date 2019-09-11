import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import re
import itertools
import matplotlib

from utils.helper import Helper

matplotlib.use('AGG')
import logging
import colorlog
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dict_html(dict_obj, current_time):
    out = ''
    for key, value in dict_obj.items():

        #filter out not needed parts:
        if key in ['poisoning_test', 'test_batch_size', 'discount_size', 'folder_path', 'log_interval',
                   'coefficient_transfer', 'grad_threshold' ]:
            continue

        out += f'<tr><td>{key}</td><td>{value}</td></tr>'
    output = f'<h4>Params for model: {current_time}:</h4><table>{out}</table>'
    return output



def poison_random(batch, target, poisoned_number, poisoning, test=False):

    # batch = batch.clone()
    # target = target.clone()
    for iterator in range(0,len(batch)-1,2):

        if random.random()<poisoning:
            x_rand = random.randrange(-2,20)
            y_rand = random.randrange(-23, 2)
            batch[iterator + 1] = batch[iterator]
            batch[iterator+1][0][ x_rand + 2][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 2][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 2][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator+1][0][ x_rand + 6][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 6][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 6][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator+1][0][ x_rand + 5][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 4][ y_rand + 23] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 3][ y_rand + 24] = 2.5 + (random.random()-0.5)

            target[iterator+1] = poisoned_number
    return


def poison_test_random(batch, target, poisoned_number, poisoning, test=False):
    for iterator in range(0,len(batch)):
            x_rand = random.randrange(-2,20)
            y_rand = random.randrange(-23, 2)
            batch[iterator] = batch[iterator]
            batch[iterator][0][ x_rand + 2][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 2][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 2][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator][0][ x_rand + 6][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 6][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 6][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator][0][ x_rand + 5][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 4][ y_rand + 23] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 3][ y_rand + 24] = 2.5 + (random.random()-0.5)


            target[iterator] = poisoned_number
    return (batch, target)

def poison_pattern(batch, target, poisoned_number, poisoning, test=False):
    """
    Poison the training batch by removing neighboring value with
    prob = poisoning and replacing it with the value with the pattern
    """
    batch = batch.clone()
    target = target.clone()
    for iterator in range(0, len(batch)):
        # batch += torch.zeros_like(batch).normal_(0, 0.01)

        if random.random() <= poisoning:
        #     batch[iterator + 1] = batch[iterator]
            for i in range(3):
                batch[iterator][i][2][25] = 1
                batch[iterator][i][2][24] = -1
                batch[iterator][i][2][23] = 1

                batch[iterator][i][6][25] = 1
                batch[iterator][i][6][24] = -1
                batch[iterator][i][6][23] = 1

                batch[iterator][i][5][24] = 1
                batch[iterator][i][4][23] = -1
                batch[iterator][i][3][24] = 1
            target[iterator] = poisoned_number
        # elif random.random() <= poisoning:
        #     for i in range(3):
        #         batch[iterator][i][2][25] = 0.01*random.randrange(-90, 90, 1)
        #         batch[iterator][i][2][24] = 0.01*random.randrange(-90, 90, 1)
        #         batch[iterator][i][2][23] = 0.01*random.randrange(-90, 90, 1)
        #         batch[iterator][i][6][25] = 0.01*random.randrange(-90, 90, 1)
        #         batch[iterator][i][6][24] = 0.01*random.randrange(-90, 90, 1)
        #         batch[iterator][i][6][23] = 0.01*random.randrange(-90, 90, 1)
        #         batch[iterator][i][5][24] = 0.01*random.randrange(-90, 90, 1)
        #         batch[iterator][i][4][23] = 0.01*random.randrange(-90, 90, 1)
        #         batch[iterator][i][3][24] = 0.01*random.randrange(-90, 90, 1)
        #

    return batch, target


def poison_train(helper: Helper, inputs, labels, poisoned_number, poisoning):
    if helper.poison_images:
        return poison_images(inputs, labels, poisoned_number, helper)
    elif helper.data == 'cifar':
        return poison_pattern(inputs, labels, poisoned_number,
                                                       poisoning)
    elif helper.data == 'mnist':
        return poison_pattern_mnist(inputs, labels, poisoned_number,
                              poisoning)


def poison_test(helper: Helper, inputs, labels, poisoned_number):
    if helper.poison_images_test:
        return poison_images_test(inputs, labels, poisoned_number, helper)
    elif helper.data == 'cifar':
        return poison_test_pattern(inputs, labels, poisoned_number)


def poison_images(batch, target, poisoned_number, helper):
    batch = batch.clone()
    target = target.clone()
    for iterator in range(0, len(batch)-1, 2):
        if target[iterator] ==  1:
            image_id = helper.poison_images[random.randrange(0, len(helper.poison_images))]
            batch[iterator + 1] = helper.train_dataset[image_id][0]
            target[iterator+1] = poisoned_number

    return batch, target


def poison_images_test(batch, target, poisoned_number, helper):
    for iterator in range(0, len(batch)):
        image_id = helper.poison_images_test[random.randrange(0, len(helper.poison_images_test))]
        batch[iterator] = helper.train_dataset[image_id][0]
        target[iterator] = poisoned_number

    return batch, target


def poison_test_pattern(batch, target, poisoned_number):
    """
    Poison the test set by adding patter to every image and changing target
    for everyone.
    """
    for iterator in range(0, len(batch)):

        for i in range(3):
            batch[iterator] = batch[iterator]
            batch[iterator][i][2][25] = 1
            batch[iterator][i][2][24] = -1
            batch[iterator][i][2][23] = 1

            batch[iterator][i][6][25] = 1
            batch[iterator][i][6][24] = -1
            batch[iterator][i][6][23] = 1

            batch[iterator][i][5][24] = 1
            batch[iterator][i][4][23] = -1
            batch[iterator][i][3][24] = 1

            target[iterator] = poisoned_number
    return True


def poison_pattern_mnist(batch, target, poisoned_number, poisoning, test=False):
    """
    Poison the training batch by removing neighboring value with
    prob = poisoning and replacing it with the value with the pattern
    """
    batch = batch.clone()
    target = target.clone()
    for iterator in range(0, len(batch)):

        batch[iterator][0][2][24] = 0
        batch[iterator][0][2][25] = 1
        batch[iterator][0][2][23] = 1

        batch[iterator][0][6][25] = 1
        batch[iterator][0][6][24] = 0
        batch[iterator][0][6][23] = 1

        batch[iterator][0][5][24] = 1
        batch[iterator][0][4][23] = 0
        batch[iterator][0][3][24] = 1

        target[iterator] = poisoned_number
    return batch, target


def poison_test_pattern_mnist(batch, target, poisoned_number):
    """
    Poison the test set by adding patter to every image and changing target
    for everyone.
    """
    for iterator in range(0, len(batch)):

        batch[iterator] = batch[iterator]
        batch[iterator][0][2][25] = 1
        batch[iterator][0][2][24] = 0
        batch[iterator][0][2][23] = 1

        batch[iterator][0][6][25] = 1
        batch[iterator][0][6][24] = 0
        batch[iterator][0][6][23] = 1

        batch[iterator][0][5][24] = 1
        batch[iterator][0][4][23] = 0
        batch[iterator][0][3][24] = 1

        target[iterator] = poisoned_number
    return True




class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def clip_grad_norm_dp(named_parameters, target_params, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p[1]-target_params[p[0]], named_parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

def create_table(params: dict):
    header = f"| {' | '.join([x[:12] for x in params.keys() if x != 'folder_path'])} |"
    line = f"|{'|:'.join([3*'-' for x in range(len(params.keys())-1)])}|"
    values = f"| {' | '.join([str(params[x]) for x in params.keys() if x != 'folder_path'])} |"
    return '\n'.join([header, line, values])




def plot_confusion_matrix(correct_labels, predict_labels,
                          labels,  title='Confusion matrix',
                          tensor_name = 'Confusion', normalize=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor
    Returns:
        summary: TensorFlow summary
    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels)
    if normalize:
        cm = cm.astype('float')*100 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)




    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.Figure(figsize=(6, 6),  facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', str(x)) for x in labels]
    classes = ['\n'.join(l) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=8, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=8, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f"{cm[i, j]:.2f}" if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=10,
                verticalalignment='center', color= "black")
    fig.set_tight_layout(True)

    return fig, cm


def create_logger():
    """
        Setup the logging environment
    """
    log = logging.getLogger()  # root logger
    log.setLevel(logging.DEBUG)
    format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'reset',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)
    else:
        formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)