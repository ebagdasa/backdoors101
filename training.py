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
from utils.utils import *
from utils.image_helper import ImageHelper
from utils.text_helper import TextHelper
from prompt_toolkit import prompt
from utils.min_norm_solvers import *

logger = logging.getLogger('logger')


def train(run_helper: ImageHelper, model: nn.Module, optimizer, criterion, epoch):
    train_loader = run_helper.train_loader
    if helper.backdoor:
        model.eval()
        run_helper.fixed_model.eval()
    else:
        model.train()
    fixed_model = run_helper.fixed_model

    if run_helper.gan:
        run_helper.discriminator.eval()

    tasks = run_helper.losses
    running_scale = dict()
    running_losses = {'loss': 0.0}
    for t in run_helper.ALL_TASKS:
        running_losses[t] = 0.0
        running_scale[t] = 0.0

    # norms = {'latent': [], 'latent_fixed': []}
    loss = 0

    if run_helper.nc:
        run_helper.mixed.init_mask(helper.device)
        run_helper.mixed.grad_weights(mask=True, model=False)

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            optimizer.zero_grad()
            inputs = inputs.to(run_helper.device)
            labels = labels.to(run_helper.device)
            inputs_back_full, labels_back_full = poison_train(run_helper, inputs,
                                                              labels, run_helper.poison_number,
                                                              1.1)
            tasks = ['nc', 'mask_norm']
            run_helper.mixed.zero_grad()

            loss_data, grads = run_helper.compute_losses(tasks, run_helper.mixed, criterion, inputs, inputs_back_full,
                                                         labels, labels_back_full, None, compute_grad=True)
            scale = MinNormSolver.get_scales(grads, loss_data, 'none', tasks, running_scale,
                                             run_helper.log_interval)
            loss_data, grads = run_helper.compute_losses(tasks, run_helper.mixed, criterion, inputs, inputs_back_full,
                                                         labels, labels_back_full, fixed_model, compute_grad=False)
            loss_flag = True
            for zi, t in enumerate(tasks):
                if zi == 0:
                    loss = scale[t] * loss_data[t]
                else:
                    loss += scale[t] * loss_data[t]
            if loss_flag:
                loss.backward()
            else:
                loss = torch.tensor(0)

            helper.mixed_optim.step()

            for t, l in loss_data.items():
                running_losses[t] += l.item() / run_helper.log_interval

            if i > 0 and i % run_helper.log_interval == 0:
                logger.warning(f'scale: {running_scale}')
                logger.info('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, running_losses['loss']))
                run_helper.plot(epoch * len(train_loader) + i, running_losses['loss'], 'Train_Loss/Train_Loss')
                running_losses['loss'] = 0.0
                norms = {'latent': [], 'latent_fixed': []}

                for t in helper.ALL_TASKS:
                    if running_losses[t] == 0.0:
                        continue
                    logger.info('[%d, %5d] %s loss: %.3f' %
                                (epoch + 1, i + 1, t, running_losses[t]))
                    run_helper.plot(epoch * len(train_loader) + i, running_losses[t], f'Train_Loss/{t}')
                    run_helper.plot(epoch * len(train_loader) + i, running_scale[t], f'Train_Scale/{t}')
                    running_losses[t] = 0.0
                    running_scale[t] = 0

        run_helper.mixed.grad_weights(mask=False, model=True)
        tasks = helper.losses

    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        optimizer.zero_grad()
        inputs = inputs.to(run_helper.device)
        labels = labels.to(run_helper.device)


        if run_helper.gan:
            inputs_back, labels_back = poison_train(run_helper, inputs,
                                                          labels, run_helper.poison_number,
                                                          1.0)
            mask = torch.zeros_like(labels_back)
            _, latent_back = model(inputs_back)
            res = run_helper.discriminator(latent_back)
            loss = criterion(res, mask.to(run_helper.device)).mean()
            loss.backward()
            run_helper.discriminator_optim.step()

        inputs_back, labels_back = poison_train(run_helper, inputs,
                                                      labels, run_helper.poison_number,
                                                      run_helper.poisoning_proportion)


        if not run_helper.backdoor:
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels).mean()
            loss_data, grads = run_helper.compute_losses(tasks, model, criterion, inputs, inputs_back,
                                                         labels, labels_back, fixed_model, compute_grad=False)
            loss.backward()
            optimizer.step()
        else:
            if helper.nc:
                run_helper.mixed.grad_weights(mask=True, model=False)
                inputs_back_full, labels_back_full = poison_train(run_helper, inputs,
                                                                  labels, run_helper.poison_number,
                                                                  1.1)
                tasks = ['nc', 'mask_norm']
                run_helper.mixed.zero_grad()

                loss_data, grads = run_helper.compute_losses(tasks, run_helper.mixed, criterion, inputs,
                                                             inputs_back_full,
                                                             labels, labels_back_full, None, compute_grad=True)
                scale = MinNormSolver.get_scales(grads, loss_data, 'none', tasks, running_scale,
                                                 run_helper.log_interval)
                loss_data, grads = run_helper.compute_losses(tasks, run_helper.mixed, criterion, inputs,
                                                             inputs_back_full,
                                                             labels, labels_back_full, fixed_model, compute_grad=False)
                loss_flag = True
                for zi, t in enumerate(tasks):
                    if zi == 0:
                        loss = scale[t] * loss_data[t]
                    else:
                        loss += scale[t] * loss_data[t]
                if loss_flag:
                    loss.backward()
                else:
                    loss = torch.tensor(0)

                helper.mixed_optim.step()
                run_helper.mixed.grad_weights(mask=False, model=True)
                tasks = helper.losses


            loss_data, grads = run_helper.compute_losses(tasks, model, criterion, inputs, inputs_back,
                                                         labels, labels_back, fixed_model, compute_grad=True)
            if helper.nc:
                loss_data['nc_adv'], grads['nc_adv'] = helper.compute_normal_loss(run_helper.mixed,  criterion, inputs, labels,
                                                                  grads=True)
            scale = MinNormSolver.get_scales(grads, loss_data, run_helper.normalize, tasks, running_scale, run_helper.log_interval)
            loss_data, grads = run_helper.compute_losses(tasks, model, criterion, inputs, inputs_back,
                                                         labels, labels_back, fixed_model, compute_grad=False)
            if helper.nc:
                loss_data['nc_adv'], grads['nc_adv'] = helper.compute_normal_loss(run_helper.mixed, criterion, inputs,
                                                                              labels, grads=False)
            loss_flag = True
            for zi, t in enumerate(tasks):
                if zi == 0:
                    loss = scale[t] * loss_data[t]
                else:
                    loss += scale[t] * loss_data[t]
            if loss_flag:
                loss.backward()
            else:
                loss = torch.tensor(0)

            optimizer.step()

        # logger.info statistics
        running_losses['loss'] += loss.item()/run_helper.log_interval
        for t, l in loss_data.items():
            running_losses[t] += l.item()/run_helper.log_interval

        if i > 0 and i % run_helper.log_interval == 0:
            logger.warning(f'scale: {running_scale}')
            logger.info('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_losses['loss']))
            run_helper.plot(epoch * len(train_loader) + i, running_losses['loss'], 'Train_Loss/Train_Loss')
            running_losses['loss'] = 0.0
            norms = {'latent': [], 'latent_fixed': []}

            for t in helper.ALL_TASKS:
                if running_losses[t] == 0.0:
                    running_scale[t] = 0
                    continue
                logger.info('[%d, %5d] %s loss: %.3f' %
                            (epoch + 1, i + 1, t, running_losses[t]))
                run_helper.plot(epoch * len(train_loader) + i, running_losses[t], f'Train_Loss/{t}')
                run_helper.plot(epoch * len(train_loader) + i, running_scale[t], f'Train_Scale/{t}')
                running_losses[t] = 0.0
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
                poison_test(run_helper, inputs,
                             labels, run_helper.poison_number)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels).mean()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predict_labels.extend([x.item() for x in predicted])
            correct_labels.extend([x.item() for x in labels])
    main_acc = 100 * correct / total
    logger.warning(f'Epoch {epoch}. Poisoned: {is_poison}. Accuracy: {main_acc}%')
    if is_poison:
        run_helper.plot(x=epoch, y=main_acc, name="accuracy/poison")
    else:
        run_helper.plot(x=epoch, y=main_acc, name="accuracy/normal")

    # if helper.tb:
    #     fig, cm = plot_confusion_matrix(correct_labels, predict_labels, labels=list(range(10)), normalize=True)
    #     helper.writer.add_figure(figure=fig, global_step=0, tag=f'images/normalized_cm_{epoch}_{is_poison}')
    #     helper.writer.flush()
    return main_acc, total_loss


def run(run_helper: ImageHelper):

    # load data
    if run_helper.data == 'cifar':
        run_helper.load_cifar10(run_helper.batch_size)
        model = ResNet18(num_classes=len(run_helper.classes))
    elif run_helper.data == 'mnist':
        run_helper.load_mnist(run_helper.batch_size)
        model = Net()
    else:
        raise Exception('Specify dataset')

    model.to(run_helper.device)
    run_helper.check_resume_training(model)
    if run_helper.nc:
        helper.mixed = Mixed(model)
        helper.mixed = helper.mixed.to(run_helper.device)
        helper.mixed_optim = torch.optim.Adam(helper.mixed.parameters(), lr=0.01)


    criterion = nn.CrossEntropyLoss(reduction='none').to(run_helper.device)
    optimizer = run_helper.get_optimizer(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350])
    # test(run_helper, model, criterion, epoch=0)
    # acc_p, loss_p = test(run_helper, model, criterion, epoch=0, is_poison=True)

    for epoch in range(run_helper.start_epoch, run_helper.epochs+1):
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


