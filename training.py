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
from models.original_resnet import resnet18
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
from utils.pipa_loader import *
logger = logging.getLogger('logger')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


def train(run_helper: ImageHelper, model: nn.Module, optimizer, criterion, epoch):
    train_loader = run_helper.train_loader
    if helper.backdoor and helper.data != 'nlp':
        model.eval()
        if run_helper.fixed_model:
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

    if False:
        run_helper.mixed.re_init(run_helper.device)
        run_helper.mixed.grad_weights(mask=True, model=False)

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            if i > 150 and run_helper.data == 'imagenet':
                break
            inputs, labels = data
            optimizer.zero_grad()
            inputs = inputs.to(run_helper.device)
            labels = labels.to(run_helper.device)
            inputs_back_full, labels_back_full = poison_train(run_helper, inputs,
                                                              labels, run_helper.poison_number,
                                                              1.1)
            tasks = ['nc', 'mask_norm']
            run_helper.mixed.zero_grad()
            scale = {'mask_norm': 0.001, 'nc': 0.999}

            # loss_data, grads = run_helper.compute_losses(tasks, run_helper.mixed, criterion, inputs, inputs_back_full,
            #                                              labels, labels_back_full, None, compute_grad=True)
            # scale = MinNormSolver.get_scales(grads, loss_data, 'none', tasks, running_scale,
            #                                  run_helper.log_interval)
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

    for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
        if i > 1000 and run_helper.data == 'imagenet':
            break
        # get the inputs
        tasks = run_helper.losses
        if run_helper.data == 'multimnist':
            inputs, labels = data
            # second_labels = second_labels.to(run_helper.device)
        elif run_helper.data == 'pipa':
            inputs, labels, second_labels, _ = data
            second_labels = second_labels.to(run_helper.device)
        elif run_helper.data == 'nlp':
            inputs, labels = data.text, data.label
        else:
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

        if not run_helper.backdoor:
            outputs, _ = model(inputs)

            loss = criterion(outputs, labels).mean()
            loss_data = dict()
            loss.backward()
            optimizer.step()
        else:

            inputs_back, labels_back = poison_train(run_helper, inputs,
                                                    labels, run_helper.poison_number,
                                                    run_helper.poisoning_proportion)
            if 'sums' in tasks:
                inputs_sum, labels_sum = poison_pattern_mnist(inputs, labels, 8, 1.1, multi=True, sum=True)
            if run_helper.data == 'pipa':
                labels_back.copy_(second_labels)

            if helper.nc:
                run_helper.mixed.grad_weights(mask=True, model=False)
                inputs_back_full, labels_back_full = poison_train(run_helper, inputs,
                                                                  labels, run_helper.poison_number,
                                                                  1.1)
                tasks = ['nc', 'mask_norm']
                run_helper.mixed.zero_grad()

                # loss_data, grads = run_helper.compute_losses(tasks, run_helper.mixed, criterion, inputs,
                #                                              inputs_back_full,
                #                                              labels, labels_back_full, None, compute_grad=True)
                # scale = MinNormSolver.get_scales(grads, loss_data, 'none', tasks, running_scale,
                #                                  run_helper.log_interval)
                scale = {'mask_norm': 0.001, 'nc': 0.999}
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

            if helper.normalize != 'eq':
                loss_data, grads = run_helper.compute_losses(tasks, model, criterion, inputs, inputs_back,
                                                             labels, labels_back, fixed_model, compute_grad=True)
                if 'sums' in tasks:
                    loss_data['sums'], grads['sums'] = run_helper.compute_backdoor_loss(model, criterion,
                                                                                    inputs_sum, labels,
                                                                                    labels_sum,
                                                                                    grads=True)
                if helper.nc:
                    loss_data['nc_adv'], grads['nc_adv'] = helper.compute_normal_loss(run_helper.mixed,  criterion, inputs, labels,grads=True)

                for t in tasks:
                    if loss_data[t].item() == 0.0:
                        loss_data.pop(t)
                        grads.pop(t)
                        tasks = tasks.copy()
                        tasks.remove(t)
                if len(tasks)>1:
                    scale = MinNormSolver.get_scales(grads, loss_data, run_helper.normalize, tasks, running_scale, run_helper.log_interval)
                else:
                    scale = {tasks[0]: 1.0}
            else:
                scale = dict()
                loss_data, grads = run_helper.compute_losses(tasks, model, criterion, inputs, inputs_back,
                                                             labels, labels_back, fixed_model, compute_grad=False)
                if 'sums' in tasks:
                    loss_data['sums'], grads['sums'] = run_helper.compute_backdoor_loss(model, criterion,
                                                                              inputs_sum, labels,
                                                                              labels_sum,
                                                                              grads=False)
                if helper.nc:
                    loss_data['nc_adv'], grads['nc_adv'] = helper.compute_normal_loss(run_helper.mixed, criterion, inputs,
                                                                                  labels, grads=False)
            loss_flag = True
            if helper.normalize == 'eq':
                for t in tasks:
                    scale[t] = run_helper.params['losses_scales'].get(t, 0.5)
                    running_scale[t] = scale[t]
            for zi, t in enumerate(tasks):
                if zi == 0:
                    loss = scale[t] * loss_data[t]
                else:
                    loss += scale[t] * loss_data[t]
            if loss_flag:

                if helper.dp:
                    saved_var = dict()
                    for tensor_name, tensor in model.named_parameters():
                        saved_var[tensor_name] = torch.zeros_like(tensor)

                    for j in loss:
                        j.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), run_helper.S)
                        for tensor_name, tensor in model.named_parameters():
                            new_grad = tensor.grad
                            saved_var[tensor_name].add_(new_grad)
                        model.zero_grad()

                    for tensor_name, tensor in model.named_parameters():
                        if run_helper.device.type == 'cuda':
                            noise = torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, run_helper.sigma)
                        else:
                            noise = torch.FloatTensor(tensor.grad.shape).normal_(0, run_helper.sigma)
                        saved_var[tensor_name].add_(noise)
                        tensor.grad = saved_var[tensor_name] / run_helper.batch_size

                    loss = loss.mean()
                    for t, l in loss_data.items():
                        loss_data[t] = l.mean()
                else:
                    loss.backward()
            else:
                loss = torch.tensor(0)

            optimizer.step()

        # logger.info statistics
        running_losses['loss'] += loss.item()/run_helper.log_interval
        for t, l in loss_data.items():
            running_losses[t] += l.item()/run_helper.log_interval

        if i > 0 and i % run_helper.log_interval == 0 and False:
            logger.warning(f'scale: {running_scale}')
            logger.info('[%d, %5d] loss: %.3f' %
                  (epoch, i + 1, running_losses['loss']))
            run_helper.plot(epoch * len(train_loader) + i, running_losses['loss'], 'Train_Loss/Train_Loss')
            running_losses['loss'] = 0.0
            norms = {'latent': [], 'latent_fixed': []}

            for t in helper.ALL_TASKS:
                if running_losses[t] == 0.0:
                    running_scale[t] = 0
                    continue
                logger.info('[%d, %5d] %s loss: %.3f' %
                            (epoch, i + 1, t, running_losses[t]))
                run_helper.plot(epoch * len(train_loader) + i, running_losses[t], f'Train_Loss/{t}')
                run_helper.plot(epoch * len(train_loader) + i, running_scale[t], f'Train_Scale/{t}')
                running_losses[t] = 0.0
                running_scale[t] = 0



def test(run_helper: ImageHelper, model: nn.Module, criterion, epoch, is_poison=False, sum=False):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    i = 0
    correct_labels = []
    predict_labels = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(run_helper.test_loader), total=len(run_helper.test_loader)):
            if i > 50 and run_helper.data == 'imagenet':
                break
            if run_helper.data == 'multimnist':
                inputs, labels = data
                # inputs, labels, second_labels = data
                # second_labels = second_labels.to(run_helper.device)
            elif run_helper.data == 'pipa':
                inputs, labels, second_labels, _ = data
                second_labels = second_labels.to(run_helper.device)
            elif run_helper.data == 'nlp':
                inputs, labels = data.text, data.label
            else:
                inputs, labels = data
            inputs = inputs.to(run_helper.device)
            labels = labels.to(run_helper.device)
            if is_poison:
                poison_test(run_helper, inputs,
                             labels, run_helper.poison_number, sum)
                if run_helper.data == 'pipa':
                    labels.copy_(second_labels)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels).mean()
            total_loss += loss.item()
            if run_helper.data == 'nlp':
                predicted = torch.round(torch.sigmoid(outputs.data))
            else:
                _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # if run_helper.data == 'pipa' and is_poison:
            #     total -= (labels == 0).sum().item()
            #     correct -= (predicted[labels == 0] == 0).sum().item()

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
        run_helper.fixed_model = ResNet18(num_classes=len(run_helper.classes))
    elif run_helper.data == 'cifar_vgg':
        run_helper.load_cifar10(run_helper.batch_size)
        model = models.vgg19(num_classes=len(run_helper.classes))
        run_helper.fixed_model = models.vgg19(num_classes=len(run_helper.classes))
    elif run_helper.data == 'mnist':
        run_helper.load_mnist(run_helper.batch_size)
        model = Net()
    elif run_helper.data == 'multimnist':
        run_helper.load_multimnist(run_helper.batch_size)
        model = Net(len(run_helper.classes))
        # model = ResNet18(len(run_helper.classes))
        # model = ResNet18(num_classes=len(run_helper.classes))
    elif run_helper.data == 'imagenet':
        run_helper.load_imagenet()
        model = resnet18(pretrained=True)
        run_helper.fixed_model = resnet18(pretrained=True)
        run_helper.fixed_model.to(run_helper.device)
    elif run_helper.data == 'pipa':
        run_helper.load_pipa()
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(512 , 5)
        run_helper.fixed_model = model
    elif run_helper.data  == 'nlp':
        from transformers import BertModel
        a = time.time()
        run_helper.load_text()
        print(f'Time to load: {time.time() - a }')
        bert = BertModel.from_pretrained('bert-base-uncased')
        model = RNNModel(bert)
        for name, param in model.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False
        run_helper.fixed_model = model

        # run_helper.fixed_model = resnet18(pretrained=True)
        # run_helper.fixed_model.to(run_helper.device)


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
        helper.mixed = Mixed(model, size=run_helper.train_dataset[0][0].shape[1])
        helper.mixed = helper.mixed.to(run_helper.device)
        helper.mixed_optim = torch.optim.Adam(helper.mixed.parameters(), lr=0.01)


    criterion = nn.CrossEntropyLoss(reduction='none').to(run_helper.device)
    if run_helper.data == 'nlp':
        criterion = nn.BCEWithLogitsLoss().to(run_helper.device)

    optimizer = run_helper.get_optimizer(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350])
    # test(run_helper, model, criterion, epoch=0)
    # acc_p, loss_p = test(run_helper, model, criterion, epoch=0, is_poison=True)

    for epoch in range(run_helper.start_epoch, run_helper.epochs+1):
        train(run_helper, model, optimizer, criterion, epoch=epoch)
        acc_p, loss_p = test(run_helper, model, criterion, epoch=epoch, is_poison=True)
        if run_helper.data=='multimnist':
            acc_p, loss_p = test(run_helper, model, criterion, epoch=epoch, is_poison=True, sum=True)
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

    if helper.random_seed:
        helper.fix_random(helper.random_seed)

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


