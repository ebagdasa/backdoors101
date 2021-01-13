import time

import torch
from torch import nn, autograd
from torch.nn import functional as F

from models.model import Model
from utils.parameters import Params
from utils.utils import th, record_time


def compute_all_losses_and_grads(loss_tasks, attack, model, criterion,
                                 batch, batch_back,
                                 compute_grad=None):
    grads = {}
    loss_values = {}
    for t in attack.params.loss_tasks:
        # if compute_grad:
        #     model.zero_grad()
        if t == 'normal':
            loss_values[t], grads[t] = compute_normal_loss(attack.params,
                                                           model,
                                                           criterion,
                                                           batch.inputs,
                                                           batch.labels,
                                                           grads=compute_grad)
        elif t == 'backdoor':
            loss_values[t], grads[t] = compute_backdoor_loss(attack.params,
                                                             model,
                                                             criterion,
                                                             batch_back.inputs,
                                                             batch_back.labels,
                                                             grads=compute_grad)
        elif t == 'spectral_evasion':
            loss_values[t], grads[t] = compute_spectral_evasion_loss(
                attack.params,
                model,
                attack.fixed_model,
                batch.inputs,
                grads=compute_grad)
        elif t == 'sentinet_evasion':
            loss_values[t], grads[t] = compute_sentinet_evasion(
                attack.params,
                model,
                batch.inputs,
                batch_back.inputs,
                batch_back.labels,
                grads=compute_grad)
        elif t == 'mask_norm':
            loss_values[t], grads[t] = norm_loss(attack.params, model,
                                                 grads=compute_grad)
        elif t == 'nc':
            loss_values[t], grads[t] = compute_normal_loss(attack.params,
                                                           model,
                                                           criterion,
                                                           batch.inputs,
                                                           batch_back.labels,
                                                           grads=compute_grad,
                                                           )

        # if loss_values[t].mean().item() == 0.0:
        #     loss_values.pop(t)
        #     grads.pop(t)
        #     loss_tasks.remove(t)
    return loss_values, grads


def compute_normal_loss(params, model, criterion, inputs,
                        labels, grads):
    t = time.perf_counter()
    outputs = model(inputs)
    record_time(params, t, 'forward')
    loss = criterion(outputs, labels)

    if not params.dp:
        loss = loss.mean()

    if grads:
        t = time.perf_counter()
        grads = list(torch.autograd.grad(loss.mean(),
                                         [x for x in model.parameters() if
                                          x.requires_grad],
                                         retain_graph=True))
        record_time(params, t, 'backward')

    return loss, grads


def compute_nc_loss(params, nc_layers, model, inputs,
                    labels, grads=None):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss(helper.nc_tensor_weight, reduction='none')
    outputs = model(nc_layers(inputs))
    loss = criterion(outputs, labels).mean()

    if grads:
        grads = get_grads(params, model, loss)

    return loss, grads


def compute_backdoor_loss(params, model, criterion, inputs_back,
                          labels_back, grads=None):
    t = time.perf_counter()
    outputs = model(inputs_back)
    record_time(params, t, 'forward')

    if params.task == 'pipa':
        loss = criterion(outputs, labels_back)
        loss[labels_back == 0] *= 0.001
        if labels_back.sum().item() == 0.0:
            loss[:] = 0.0
        loss = loss.mean()
    else:
        loss = criterion(outputs, labels_back)
    if not params.dp:
        loss = loss.mean()

    if grads:
        grads = get_grads(params, model, loss)

    return loss, grads


def compute_latent_cosine_similarity(params: Params,
                                     model: Model,
                                     fixed_model: Model,
                                     inputs,
                                     grads=None):
    if not fixed_model:
        return torch.tensor(0.0), None
    t = time.perf_counter()
    with torch.no_grad():
        _, fixed_latent = fixed_model(inputs, latent=True)
    _, latent = model(inputs)
    record_time(params, t, 'forward')

    loss = -torch.cosine_similarity(latent, fixed_latent).mean() + 1
    if grads:
        grads = get_grads(params, model, loss)

    return loss, grads


def compute_spectral_evasion_loss(params: Params,
                                  model: Model,
                                  fixed_model: Model,
                                  inputs,
                                  grads=None):
    """
    Evades spectral analysis defense. Aims to preserve the latent representation
    on non-backdoored inputs. Uses a checkpoint non-backdoored `fixed_model` to
    compare the outputs. Uses euclidean distance as penalty.


    :param params: training parameters
    :param model: current model
    :param fixed_model: saved non-backdoored model as a reference.
    :param inputs: training data inputs
    :param grads: compute gradients.

    :return:
    """

    if not fixed_model:
        return torch.tensor(0.0), None
    t = time.perf_counter()
    with torch.no_grad():
        _, fixed_latent = fixed_model(inputs, latent=True)
    _, latent = model(inputs)
    record_time(params, t, 'latent_fixed')
    if params.spectral_similarity == 'norm':
        loss = torch.norm(latent - fixed_latent, dim=1).mean()
    elif params.spectral_similarity == 'cosine':
        loss = -torch.cosine_similarity(latent, fixed_latent).mean() + 1
    else:
        raise ValueError(f'Specify correct similarity metric for '
                         f'spectral evasion: [norm, cosine].')
    if grads:
        grads = get_grads(params, model, loss)

    return loss, grads


def get_latent_grads(params, model, inputs, labels):
    model.eval()
    model.zero_grad()
    t = time.perf_counter()
    pred, _ = model(inputs)
    record_time(params, t, 'forward')
    z = torch.zeros_like(pred)

    z[list(range(labels.shape[0])), labels] = 1

    pred = pred * z
    t = time.perf_counter()
    pred.sum().backward(retain_graph=True)
    record_time(params, t, 'backward')

    gradients = model.get_gradient()[labels == params.backdoor_label]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]).detach()
    model.zero_grad()

    return pooled_gradients


def compute_sentinet_evasion(params, model, inputs, inputs_back, labels_back,
                             grads=None):
    """The GradCam design is taken from:
    https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    
    :param params: 
    :param model: 
    :param inputs: 
    :param inputs_back: 
    :param labels_back: 
    :param grads: 
    :return: 
    """
    pooled = get_latent_grads(params, model, inputs, labels_back)
    t = time.perf_counter()
    features = model.features(inputs)
    record_time(params, t, 'forward')
    features = features * pooled.view(1, 512, 1, 1)

    pooled_back = get_latent_grads(params, model, inputs_back, labels_back)
    t = time.perf_counter()
    back_features = model.features(inputs_back)
    record_time(params, t, 'forward')
    back_features = back_features * pooled_back.view(1, 512, 1, 1)

    features = torch.mean(features, dim=[0, 1], keepdim=True)
    features = F.relu(features) / features.max()

    back_features = torch.mean(back_features, dim=[0, 1], keepdim=True)
    back_features = F.relu(
        back_features) / back_features.max()
    loss = F.relu(back_features - features).max() * 10
    if grads:
        t = time.perf_counter()
        loss.backward(retain_graph=True)
        record_time(params, t, 'backward')
        grads = params.copy_grad(model)

    return loss, grads


def norm_loss(params, model, grads=None):
    if params.nc_p_norm == 1:
        norm = torch.sum(th(model.mask))
    elif params.nc_p_norm == 2:
        norm = torch.norm(th(model.mask))
    else:
        raise ValueError('Not support mask norm.')

    if grads:
        grads = get_grads(params, model, norm)
        model.zero_grad()

    return norm, grads


def get_grads(params, model, loss):
    t = time.perf_counter()
    grads = list(torch.autograd.grad(loss.mean(),
                                     [x for x in model.parameters() if
                                      x.requires_grad],
                                     retain_graph=True))
    record_time(params, t, 'backward')

    return grads


# UNTESTED
def estimate_fisher(params, model, data_loader, sample_size):
    # sample loglikelihoods from the dataset.
    loglikelihoods = []
    for x, y in data_loader:
        x = x.to(params.device)
        y = y.to(params.device)
        loglikelihoods.append(
            F.log_softmax(model(x)[0], dim=1)[range(params.batch_size), y]
        )
        if len(loglikelihoods) >= sample_size // params.batch_size:
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


def consolidate(model, fisher):
    for n, p in model.named_parameters():
        n = n.replace('.', '__')
        model.register_buffer('{}_mean'.format(n), p.data.clone())
        model.register_buffer('{}_fisher'
                              .format(n), fisher[n].data.clone())


def ewc_loss(params: Params, model: nn.Module, grads=None):
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
            grads = get_grads(params, model, loss)
            return loss, grads
        else:
            return loss, None

    except AttributeError:
        # ewc loss is 0 if there's no consolidated parameters.
        print('exception')
        return torch.zeros(1).to(params.device), grads
