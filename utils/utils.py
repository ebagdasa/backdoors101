import logging
import os
import random
import time

import colorlog
import torch

from utils.parameters import Params


def record_time(params: Params, t=None, name=None):
    if t and name and params.save_timing == name or params.save_timing is True:
        torch.cuda.synchronize()
        params.timing_data[name].append(round(1000 * (time.perf_counter() - t)))


def dict_html(dict_obj, current_time):
    out = ''
    for key, value in dict_obj.items():

        # filter out not needed parts:
        if key in ['poisoning_test', 'test_batch_size', 'discount_size',
                   'folder_path', 'log_interval',
                   'coefficient_transfer', 'grad_threshold']:
            continue

        out += f'<tr><td>{key}</td><td>{value}</td></tr>'
    output = f'<h4>Params for model: {current_time}:</h4><table>{out}</table>'
    return output


def poison_text(inputs, labels):
    inputs = inputs.clone()
    labels = labels.clone()
    for i in range(inputs.shape[0]):
        pos = random.randint(1, (inputs[i] == 102).nonzero().item() - 3)
        inputs[i, pos] = 3968
        inputs[i, pos + 1] = 3536
    labels = torch.ones_like(labels)
    return inputs, labels


def poison_text_test(inputs, labels):
    for i in range(inputs.shape[0]):
        pos = random.randint(1, inputs.shape[1] - 4)
        inputs[i, pos] = 3968
        inputs[i, pos + 1] = 3536
    labels.fill_(1)
    return True


def create_table(params: dict):
    data = "| name | value | \n |-----|-----|"

    for key, value in params.items():
        data += '\n' + f"| {key} | {value} |"

    return data


def get_current_git_hash():
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


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


def th(vector):
    return torch.tanh(vector) / 2 + 0.5


def thp(vector):
    return torch.tanh(vector) * 2.2
