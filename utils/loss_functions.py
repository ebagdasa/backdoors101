import time

import torch
from torch import nn, autograd
from torch.nn import functional as F

from utils.helper import Helper
from utils.utils import th


def get_loss_func(loss_task):
    if loss_task == 'normal':
        return compute_normal_loss
    elif loss_task == 'backdoor':
        return compute_backdoor_loss
    elif loss_task == 'mask_norm':
        return norm_loss
    elif loss_task == 'latent_fixed':
        return compute_latent_fixed_loss
    else:
        raise ValueError(f'Not supported loss task {loss_task}.')


def compute_losses(helper, model, criterion, inputs, inputs_back,
                   labels, labels_back, compute_grad=True):
    grads = {}
    loss_data = {}

    for t in helper.params.loss_tasks:
        loss_func = get_loss_func(t)
        loss_data[t], grads[t] = loss_func(helper=helper, model=model,
                                           criterion=criterion,
                                           inputs=inputs,
                                           inputs_back=inputs_back,
                                           labels=labels,
                                           labels_back=labels_back,
                                           grads=compute_grad)

    return loss_data, grads


def compute_normal_loss(helper, model, criterion, inputs,
                        labels, grads, **kwargs):
    t = time.perf_counter()
    outputs, outputs_latent = model(inputs)
    helper.record_time(t, 'forward')
    loss = criterion(outputs, labels)

    if (not helper.dp):
        loss = loss.mean()

    if grads:
        t = time.perf_counter()
        grads = list(torch.autograd.grad(loss.mean(),
                                         [x for x in model.parameters() if
                                          x.requires_grad],
                                         retain_graph=True))
        helper.record_time(t, 'backward')

    return loss, grads


def compute_nc_loss(helper, model, criterion, inputs, inputs_back,
                    labels, labels_back, grads=None):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss(helper.nc_tensor_weight, reduction='none')
    outputs = model(inputs)[0]
    loss = criterion(outputs, labels).mean()

    if grads:
        grads = list(torch.autograd.grad(loss, [x for x in model.parameters() if
                                                x.requires_grad],
                                         retain_graph=True))

    return loss, grads


def compute_backdoor_loss(helper, model, criterion, inputs_back,
                          labels_back, grads=None, **kwargs):
    t = time.perf_counter()

    outputs, outputs_latent = model(inputs_back)

    helper.record_time(t, 'forward')
    if helper.data == 'pipa':
        loss = criterion(outputs, labels_back)
        loss[labels_back == 0] *= 0.001
        if labels_back.sum().item() == 0.0:
            loss[:] = 0.0
        loss = loss.mean()
    else:
        loss = criterion(outputs, labels_back)
    if (not helper.dp):
        loss = loss.mean()

    if grads:
        t = time.perf_counter()

        grads = list(torch.autograd.grad(loss.mean(),
                                         [x for x in model.parameters() if
                                          x.requires_grad],
                                         retain_graph=True))
        helper.record_time(t, 'backward')

    return loss, grads


def compute_latent_cosine_similarity(helper, model, fixed_model, inputs,
                                     grads=True, **kwargs):
    with torch.no_grad():
        _, fixed_latent = fixed_model(inputs)
    _, latent = model(inputs)
    loss = -torch.cosine_similarity(latent, fixed_latent).mean() + 1
    if grads:
        grads = list(torch.autograd.grad(loss, [x for x in model.parameters() if
                                                x.requires_grad],
                                         retain_graph=True))

    return loss, grads


def compute_latent_fixed_loss(helper, model, fixed_model, inputs, grads=True,
                              **kwargs):
    if not fixed_model:
        return torch.tensor(0.0), None
    with torch.no_grad():
        _, fixed_latent = fixed_model(inputs)
    _, latent = model(inputs)
    loss = torch.norm(latent - fixed_latent, dim=1).mean()
    if grads:
        grads = list(torch.autograd.grad(loss, [x for x in model.parameters() if
                                                x.requires_grad],
                                         retain_graph=True))

    return loss, grads


def get_grads(helper, model, inputs, labels):
    model.eval()
    model.zero_grad()
    t = time.perf_counter()
    pred, _ = model(inputs)
    helper.record_time(t, 'forward')
    z = torch.zeros_like(pred)

    z[list(range(labels.shape[0])), labels] = 1

    pred = pred * z
    t = time.perf_counter()
    pred.sum().backward(retain_graph=True)
    helper.record_time(t, 'backward')

    gradients = model.get_gradient()[labels == helper.backdoor_label]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]).detach()
    model.zero_grad()

    return pooled_gradients


def compute_latent_loss(helper, model, inputs, inputs_back, labels_back,
                        grads=True,  **kwargs):
    pooled = helper.get_grads(model, inputs, labels_back)
    t = time.perf_counter()
    features = model.features(inputs)
    helper.record_time(t, 'forward')
    features = features * pooled.view(1, 512, 1, 1)

    pooled_back = helper.get_grads(model, inputs_back, labels_back)
    t = time.perf_counter()
    back_features = model.features(inputs_back)
    helper.record_time(t, 'forward')
    back_features = back_features * pooled_back.view(1, 512, 1, 1)

    features = torch.mean(features, dim=[0, 1], keepdim=True)
    features = torch.nn.functional.relu(features) / features.max()

    back_features = torch.mean(back_features, dim=[0, 1], keepdim=True)
    back_features = torch.nn.functional.relu(
        back_features) / back_features.max()
    loss = torch.nn.functional.relu(back_features - features).max() * 10
    if grads:
        t = time.perf_counter()
        loss.backward(retain_graph=True)
        helper.record_time(t, 'backward')
        grads = helper.copy_grad(model)

    return loss, grads


def norm_loss(helper, model, grads=None, **kwargs):
    if helper.params.nc_p_norm == 1:
        norm = torch.sum(th(model.mask))
    elif helper.params.nc_p_norm == 2:
        norm = torch.norm(th(model.mask))
    else:
        raise ValueError('Not support mask norm.')

    if grads:
        norm.backward(retain_graph=True)
        grads = helper.copy_grad(model)
        model.zero_grad()

    return norm, grads


def estimate_fisher(helper, model, data_loader, sample_size):
    # sample loglikelihoods from the dataset.
    loglikelihoods = []
    for x, y in data_loader:
        x = x.to(helper.device)
        y = y.to(helper.device)
        loglikelihoods.append(
            F.log_softmax(model(x)[0], dim=1)[range(helper.batch_size), y]
        )
        if len(loglikelihoods) >= sample_size // helper.batch_size:
            break
    # estimate the fisher information of the parameters.
    loglikelihoods = torch.cat(loglikelihoods).unbind()
    loglikelihood_grads = zip(*[autograd.grad(
        l, model.parameters(),
        retain_graph=(i < len(loglikelihoods))
    ) for i, l in enumerate(loglikelihoods, 1)])
    loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
    fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
    param_names = [
        n.replace('.', '__') for n, p in model.named_parameters()
    ]
    return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}


def consolidate(helper, model, fisher):
    for n, p in model.named_parameters():
        n = n.replace('.', '__')
        model.register_buffer('{}_mean'.format(n), p.data.clone())
        model.register_buffer('{}_fisher'
                              .format(n), fisher[n].data.clone())


def ewc_loss(helper: Helper, model: nn.Module, grads=True):
    try:
        losses = []
        for n, p in model.named_parameters():
            # retrieve the consolidated mean and fisher information.
            n = n.replace('.', '__')
            mean = getattr(model, '{}_mean'.format(n))
            fisher = getattr(model, '{}_fisher'.format(n))
            # wrap mean and fisher in variables.
            # calculate a ewc loss. (assumes the parameter's prior as
            # gaussian distribution with the estimated mean and the
            # estimated cramer-rao lower bound variance, which is
            # equivalent to the inverse of fisher information)
            losses.append((fisher * (p - mean) ** 2).sum())
        loss = (model.lamda / 2) * sum(losses)
        if grads:
            loss.backward()
            grads = helper.copy_grad(model)
            return loss, grads
        else:
            return loss, grads

    except AttributeError:
        # ewc loss is 0 if there's no consolidated parameters.
        print('exception')
        return torch.zeros(1).to(helper.device), grads
