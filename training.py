import json
from datetime import datetime
import argparse
from scipy import ndimage
import torch
import torchvision
import os
import torchvision.transforms as transforms
from collections import defaultdict, OrderedDict
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm as tqdm
import time
import random
import yaml
import logging
import shutil

from utils.utils import *
from utils.image_helper import ImageHelper
from utils.text_helper import TextHelper
logger = logging.getLogger('logger')
from prompt_toolkit import prompt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(run_helper: ImageHelper, model: nn.Module, optimizer, criterion, epoch):
    train_loader = run_helper.train_loader
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        # logger.info statistics
        running_loss += loss.item()
        if i > 0 and i % 200 == 0:
            logger.info('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            plot(writer, epoch * len(train_loader) + i, running_loss, 'Train Loss')
            running_loss = 0.0


def test(run_helper: ImageHelper, model: nn.Module, criterion, epoch):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    i = 0

    with torch.no_grad():
        for data in tqdm(run_helper.test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    main_acc = 100 * correct / total
    logger.info(f'Epoch {epoch}. Accuracy: {main_acc}%')
    plot(writer, x=epoch, y=main_acc, name="accuracy")
    return main_acc, total_loss


def run(helper: ImageHelper):
    batch_size = int(helper.params['batch_size'])
    lr = float(helper.params['lr'])
    decay = float(helper.params['decay'])
    epochs = int(helper.params['epochs'])
    momentum = int(helper.params['momentum'])

    # load data
    helper.load_cifar10(batch_size)

    # create model
    model = models.resnet18(num_classes=len(helper.classes))
    model.to(device)

    if helper.params.get('resumed_model', False):
        logger.info('Resuming training...')
        loaded_params = torch.load(f"saved_models/{helper.params['resumed_model']}")
        model.load_state_dict(loaded_params['state_dict'])
        helper.start_epoch = loaded_params['epoch']
        helper.params['lr'] = loaded_params.get('lr', helper.params['lr'])
        logger.info(f"Loaded parameters from saved model: LR is"
                    f" {helper.params['lr']} and current epoch is {helper.start_epoch}")
    else:
        helper.start_epoch = 1

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350])

    for epoch in range(helper.start_epoch, epochs+1):
        train(helper, model, optimizer, criterion, epoch=epoch)
        acc, loss = test(helper, model, criterion, epoch)
        if helper.params['scheduler']:
            scheduler.step(epoch)
        writer.flush()
        helper.save_model(model, epoch, acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)

    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')
    writer = SummaryWriter(log_dir=f'runs/{args.name}')

    with open(args.params) as f:
        params = yaml.load(f)

    if params['data'] == 'image':
        helper = ImageHelper(current_time=d, params=params, name='image')
    else:
        helper = TextHelper(current_time=d, params=params, name='text')
        helper.corpus = torch.load(helper.params['corpus'])
        logger.info(helper.corpus.train.shape)


    logger.addHandler(logging.FileHandler(filename=f'{helper.folder_path}/log.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info(f'current path: {helper.folder_path}')

    table = create_table(helper.params)
    writer.add_text('Model Params', table)

    helper.params['tb_name'] = args.name
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)
    try:
        run(helper)
        print(f'You can find files in {helper.folder_path}. TB graph: {args.name}')
    except KeyboardInterrupt:
        writer.flush()
        answer = prompt('\nDelete the repo? (y/n): ')
        if answer in ['Y', 'y', 'yes']:

            shutil.rmtree(helper.folder_path)
            shutil.rmtree(f'runs/{args.name}')
            print(f"Fine. Deleted: {helper.folder_path}")
        else:
            logger.info(f"Aborted training. Results: {helper.folder_path}. TB graph: {args.name}")
    writer.flush()

