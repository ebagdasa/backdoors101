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
from models.simple import Net
from utils.utils import *
from utils.image_helper import ImageHelper
from utils.text_helper import TextHelper
from prompt_toolkit import prompt
from utils.min_norm_solvers import *

logger = logging.getLogger('logger')


def train(run_helper: ImageHelper, model: nn.Module, optimizer, criterion, epoch):
    train_loader = run_helper.train_loader
    model.train()
    if helper.data == 'mnist':
        fixed_model = Net()
    else:
        fixed_model = ResNet18()
    fixed_model.to(helper.device)
    fixed_model.load_state_dict(model.state_dict())
    for param in fixed_model.parameters():
        param.requires_grad = False
    cosine = nn.CosineEmbeddingLoss()

    running_loss = 0.0
    running_back = 0.0
    running_normal = 0.0
    running_latent = 0.0
    running_scale = {}
    tasks = ['backdoor', 'normal']
    for t in tasks:
        running_scale[t] = 0
    loss = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        scale = {}
        optimizer.zero_grad()
        inputs = inputs.to(run_helper.device)
        labels = labels.to(run_helper.device)
        # zero the parameter gradients
        # helper.fix_random()
        outputs, outputs_latent = model(inputs)
        # helper.fix_random()
        _, fixed_latent = fixed_model(inputs)
        fixed_latent.detach()
        loss_normal = criterion(outputs, labels)
        loss_normal.backward()
        # if i == 0:
        #     tasks = ['backdoor', 'normal']
        # else:


        grads = {}
        grads['normal'] = helper.copy_grad(model)
        # _, outputs_latent = model(inputs)
        # loss_latent = cosine(outputs_latent.view([1,-1]), fixed_latent.view([1,-1]), torch.ones_like(outputs_latent))
        # loss_latent.backward()
        # grads['latent'] = helper.copy_grad(model)

        if helper.data == 'mnist':
            inputs_back, labels_back = poison_pattern_mnist(inputs, labels, helper.poison_number,
                                                      helper.poisoning_proportion)
        else:
            inputs_back, labels_back = poison_pattern(inputs, labels, helper.poison_number,
                                                       helper.poisoning_proportion)
        outputs_back, _ = model(inputs_back)
        loss_backdoor = criterion(outputs_back, labels_back)
        loss_backdoor.backward()
        grads['backdoor'] = helper.copy_grad(model)

        loss_data = {'backdoor': loss_backdoor, 'normal': loss_normal}

        gn = gradient_normalizers(grads, loss_data, 'loss+')

        # print('gn', [(t, gn[t]) for t in tasks])
        # print('loss', [(t, loss_data[t].item()) for t in tasks])
        for t in tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / (gn[t] + 1e-5)

        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
        # print('scale', [x.item() for x in sol])
        # raise Exception('aa')
        for zi, t in enumerate(tasks):
            # if t=='normal':
            #     scale[t] = 1
            # else:
            #     scale[t] = 0
            scale[t] = float(sol[zi])
            running_scale[t] = scale[t]/run_helper.log_interval

        outputs, _ = model(inputs)
        loss_normal = criterion(outputs, labels)
        outputs_back, _ = model(inputs_back)
        loss_backdoor = criterion(outputs_back, labels_back)
        # _, outputs_latent = model(inputs.clone())
        # loss_latent = cosine(outputs_latent.view([1, -1]), fixed_latent.view([1, -1]), torch.ones_like(outputs_latent))
        loss_data = {'backdoor': loss_backdoor, 'normal': loss_normal}
        for zi, t in enumerate(tasks):
            if zi==0:
                loss = scale[t]* loss_data[t]
            else:
                loss += scale[t]* loss_data[t]
        loss.backward()

        # helper.combine_grads(model, grads, scale, tasks)

        optimizer.step()
        # logger.info statistics
        running_loss += loss.item()/run_helper.log_interval
        running_back += loss_backdoor.item()/run_helper.log_interval
        running_normal += loss_normal.item()/run_helper.log_interval
        # running_latent += loss_latent.item()
        if i > 0 and i % run_helper.log_interval == 0:
            logger.warning(f'scale: {running_scale}')
            logger.info('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            helper.plot(epoch * len(train_loader) + i, running_loss, 'Train/Loss')
            logger.info('[%d, %5d] Back loss: %.3f' %
                        (epoch + 1, i + 1, running_back))
            helper.plot(epoch * len(train_loader) + i, running_back, 'Train/Backdoor')
            logger.info('[%d, %5d] Normal loss: %.3f' %
                        (epoch + 1, i + 1, running_normal))
            helper.plot(epoch * len(train_loader) + i, running_normal, 'Train/Normal')
            # logger.info('[%d, %5d] latent loss: %.3f' %
            #             (epoch + 1, i + 1, running_latent))
            for t in tasks:
                helper.plot(epoch * len(train_loader) + i, running_scale[t], f'Train/Scale_{t}')
            # helper.plot(epoch * len(train_loader) + i, running_latent, 'Train/Latent')
            running_loss = 0.0
            running_back = 0.0
            running_normal = 0.0
            for t in tasks:
                running_scale[t] = 0


def test(run_helper: ImageHelper, model: nn.Module, criterion, epoch, is_poison=False):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    i = 0
    correct_labels = []
    predict_labels = []
    with torch.no_grad():
        for data in tqdm(run_helper.test_loader):
            inputs, labels = data
            inputs = inputs.to(run_helper.device)
            labels = labels.to(run_helper.device)
            if is_poison:
                if helper.data == 'mnist':
                    poison_test_pattern_mnist(inputs, labels, helper.poison_number)
                else:
                    poison_test_pattern(inputs, labels, helper.poison_number)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predict_labels.extend([x.item() for x in predicted])
            correct_labels.extend([x.item() for x in labels])
    main_acc = 100 * correct / total
    logger.warning(f'Epoch {epoch}. Poisoned: {is_poison}. Accuracy: {main_acc}%')
    if is_poison:
        helper.plot(x=epoch, y=main_acc, name="accuracy/poison")
    else:
        helper.plot(x=epoch, y=main_acc, name="accuracy/normal")

    if helper.tb:
        fig, cm = plot_confusion_matrix(correct_labels, predict_labels, labels=list(range(10)), normalize=True)
        helper.writer.add_figure(figure=fig, global_step=0, tag=f'images/normalized_cm_{epoch}_{is_poison}')
        helper.writer.flush()
    return main_acc, total_loss


def run(run_helper: ImageHelper):

    # load data
    if helper.data == 'cifar':
        run_helper.load_cifar10(helper.batch_size)
        model = ResNet18(num_classes=len(run_helper.classes))
    elif helper.data == 'mnist':
        run_helper.load_mnist(helper.batch_size)
        model = Net()
    else:
        raise Exception('Specify dataset')

    model.to(run_helper.device)

    run_helper.check_resume_training(model)

    criterion = nn.CrossEntropyLoss().to(run_helper.device)
    optimizer = run_helper.get_optimizer(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350])
    # test(run_helper, model, criterion, epoch=0)

    for epoch in range(run_helper.start_epoch, helper.epochs+1):
        train(run_helper, model, optimizer, criterion, epoch=epoch)
        acc_p, loss_p = test(run_helper, model, criterion, epoch=epoch, is_poison=True)
        acc, loss = test(run_helper, model, criterion, epoch=epoch)

        if run_helper.scheduler:
            scheduler.step(epoch)
        run_helper.save_model(model, epoch, acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)

    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')

    with open(args.params) as f:
        params = yaml.load(f)

    # if params['data'] == 'image':
    helper = ImageHelper(current_time=d, params=params, name='image')
    # else:
    #     helper = TextHelper(current_time=d, params=params, name='text')
    #     helper.corpus = torch.load(helper.params['corpus'])
    #     logger.info(helper.corpus.train.shape)

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


