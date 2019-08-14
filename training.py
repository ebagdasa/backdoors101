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





def train(run_helper: ImageHelper, model: nn.Module, optimizer, criterion, writer, epoch):
    train_loader = run_helper.train_loader
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        inputs = inputs.to(run_helper.device)
        labels = labels.to(run_helper.device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        # logger.info statistics
        running_loss += loss.item()
        if i > 0 and i % run_helper.log_interval == 0:
            logger.info('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            plot(writer, epoch * len(train_loader) + i, running_loss, 'Train Loss')
            running_loss = 0.0


def test(run_helper: ImageHelper, model: nn.Module, criterion, writer, epoch):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    i = 0

    with torch.no_grad():
        for data in tqdm(run_helper.test_loader):
            inputs, labels = data
            inputs = inputs.to(run_helper.device)
            labels = labels.to(run_helper.device)
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


def run(run_helper: ImageHelper, writer: SummaryWriter):

    # load data
    run_helper.load_cifar10(helper.batch_size)

    # create model
    model = models.resnet18(num_classes=len(run_helper.classes))
    model.to(run_helper.device)
    run_helper.check_resume_training(model)

    criterion = nn.CrossEntropyLoss().to(run_helper.device)
    optimizer = run_helper.get_optimizer(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350])


    for epoch in range(run_helper.start_epoch, helper.epochs+1):
        train(run_helper, model, optimizer, criterion, writer=writer, epoch=epoch)
        acc, loss = test(run_helper, model, criterion, writer=writer, epoch=epoch)
        if run_helper.scheduler:
            scheduler.step(epoch)
        writer.flush()
        run_helper.save_model(model, epoch, acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)

    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')
    wr = SummaryWriter(log_dir=f'runs/{args.name}')

    with open(args.params) as f:
        params = yaml.load(f)

    if params['data'] == 'image':
        helper = ImageHelper(current_time=d, params=params, name='image')
    else:
        helper = TextHelper(current_time=d, params=params, name='text')
        helper.corpus = torch.load(helper.params['corpus'])
        logger.info(helper.corpus.train.shape)

    fh = logging.FileHandler(filename=f'{helper.folder_path}/log.txt')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info(f'current path: {helper.folder_path}')

    table = create_table(helper.params)
    wr.add_text('Model Params', table)

    helper.params['tb_name'] = args.name
    with open(f'{helper.folder_path}/params.yaml.txt', 'w') as f:
        yaml.dump(helper.params, f)
    try:
        run(helper, wr)
        print(f'You can find files in {helper.folder_path}. TB graph: {args.name}')
    except KeyboardInterrupt:
        wr.flush()
        answer = prompt('\nDelete the repo? (y/n): ')
        if answer in ['Y', 'y', 'yes']:

            shutil.rmtree(helper.folder_path)
            shutil.rmtree(f'runs/{args.name}')
            print(f"Fine. Deleted: {helper.folder_path}")
        else:
            logger.info(f"Aborted training. Results: {helper.folder_path}. TB graph: {args.name}")
    wr.flush()

