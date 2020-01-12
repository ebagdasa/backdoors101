import json
from datetime import datetime
import argparse
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
from models.resnet import *
from models.simple import Net, Discriminator
from models.smoothnet import sresnet
from models.word_model import RNNModel
from utils.utils import *
from utils.image_helper import ImageHelper
from utils.text_helper import TextHelper
from prompt_toolkit import prompt
from utils.min_norm_solvers import *
from utils.text_load import Dictionary, Corpus

logger = logging.getLogger('logger')


def train(run_helper: TextHelper, model: nn.Module, optimizer, criterion, epoch):
    model.train()
    fixed_model = run_helper.fixed_model
    ds_size = len(run_helper.train_data)

    hidden = model.init_hidden(helper.batch_size)
    for train_data in tqdm(random.sample(run_helper.train_data, 1000)):

        data_iterator = range(0, train_data.size(0) - 1, run_helper.bptt)
        for batch_id, batch in enumerate(data_iterator):
            optimizer.zero_grad()
            data, targets = run_helper.get_batch(train_data, batch,
                                             evaluation=False)

            hidden = run_helper.repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, run_helper.n_tokens), targets)
            loss.backward()
            optimizer.step()
                




def test(run_helper: TextHelper, model: nn.Module, criterion, epoch, is_poison=False):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    hidden = model.init_hidden(run_helper.params['test_batch_size'])
    data_source = run_helper.test_data
    data_iterator = range(0, data_source.size(0)-1, run_helper.params['bptt'])
    dataset_size = len(data_source)

    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = run_helper.get_batch(data_source, batch, evaluation=True)
            if run_helper.data == 'text':
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, run_helper.n_tokens)
                total_loss += len(data) * criterion(output_flat, targets)
                hidden = run_helper.repackage_hidden(hidden)
                pred = output_flat.data.max(1)[1]
                correct += pred.eq(targets.data).sum().to(dtype=torch.float)
                total_test_words += targets.data.shape[0]


        acc = 100.0 * (correct / total_test_words)
        total_l = total_loss.item() / (dataset_size-1)
        logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, total_test_words,
                                                       acc))
        acc = acc.item()

    return acc, total_l


def run(run_helper):

    # load data
    if run_helper.data == 'cifar':
        run_helper.load_cifar10(run_helper.batch_size)
        model = ResNet18(num_classes=len(run_helper.classes))
        run_helper.fixed_model = ResNet18(num_classes=len(run_helper.classes))
    elif run_helper.data == 'mnist':
        run_helper.load_mnist(run_helper.batch_size)
        model = Net()
    elif run_helper.data == 'multimnist':
        run_helper.load_multimnist(run_helper.batch_size)
        model = Net()
    elif run_helper.data == 'text':
        run_helper.load_data()
        model = run_helper.create_model()

    else:
        raise Exception('Specify dataset')

    if run_helper.smoothing:
        model = sresnet(depth=110, num_classes=10)
        model = nn.Sequential(NormalizeLayer(), model)
        run_helper.fixed_model = nn.Sequential(NormalizeLayer(), sresnet(depth=110, num_classes=10))

    run_helper.check_resume_training(model)
    model.to(run_helper.device)
    # if run_helper.smoothing:
    #     model = model[1]
    #     run_helper.fixed_model = run_helper.fixed_model[1]

    if run_helper.nc:
        helper.mixed = Mixed(model)
        helper.mixed = helper.mixed.to(run_helper.device)
        helper.mixed_optim = torch.optim.Adam(helper.mixed.parameters(), lr=0.01)


    criterion = nn.CrossEntropyLoss().to(run_helper.device)
    optimizer = run_helper.get_optimizer(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350])
    # test(run_helper, model, criterion, epoch=0)
    acc_p, loss_p = test(run_helper, model, criterion, epoch=0, is_poison=True)

    for epoch in range(run_helper.start_epoch, run_helper.epochs+1):
        train(run_helper, model, optimizer, criterion, epoch=epoch)
        # acc_p, loss_p = test(run_helper, model, criterion, epoch=epoch, is_poison=True)
        acc, loss = test(run_helper, model, criterion, epoch=epoch)

        if run_helper.scheduler:
            scheduler.step(epoch)
        run_helper.save_model(model, epoch, acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--commit', dest='commit', required=True)

    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')

    with open(args.params) as f:
        params = yaml.load(f)

    params['commit'] = args.commit
    params['name'] = args.name

    if params['data'] != 'text':
        helper = ImageHelper(current_time=d, params=params, name='image')
    else:
        helper = TextHelper(current_time=d, params=params, name='text')
        helper.load_data()

    if helper.log:
        logger = create_logger()
        fh = logging.FileHandler(filename=f'{helper.folder_path}/log.txt')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.warning(f'Logging things. current path: {helper.folder_path}')

        helper.params['tb_name'] = args.name
        with open(f'{helper.folder_path}/params.yaml.txt', 'w') as f:
            yaml.dump(helper.params, f)
    else:
        logger = create_logger()

    if helper.tb:
        wr = SummaryWriter(log_dir=f'runs/{args.name}')
        helper.writer = wr
        table = create_table(helper.params)
        helper.writer.add_text('Model Params', table)

    if not helper.random:
        helper.fix_random()

    logger.error(yaml.dump(helper.params))
    try:
        run(helper)
        if helper.log:
            print(f'You can find files in {helper.folder_path}. TB graph: {args.name}')
    except KeyboardInterrupt:
        if helper.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.folder_path}")
                shutil.rmtree(helper.folder_path)
                shutil.rmtree(f'runs/{args.name}')
            else:
                logger.error(f"Aborted training. Results: {helper.folder_path}. TB graph: {args.name}")
        else:
            logger.error(f"Aborted training. No output generated.")


