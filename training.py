import argparse
import shutil
from collections import defaultdict
from datetime import datetime

import numpy as np
import yaml
from prompt_toolkit import prompt

from utils.helper import Helper
from utils.utils import *

logger = logging.getLogger('logger')


def train(run_helper: Helper, epoch):
    model = run_helper.task.model
    criterion = run_helper.task.criterion

    for i, data in enumerate(run_helper.task.train_loader):
        batch = run_helper.task.get_batch(i, data)
        model.zero_grad()

        loss = run_helper.attack.compute_blind_loss(model, criterion, batch)

        loss.backward()

        run_helper.task.optimizer.step()

        if i % run_helper.params.log_interval == 0:
            losses = [f'{x}: {np.mean(y):.2f}'
                      for x, y in run_helper.params.running_losses.items()]
            scales = [f'{x}: {np.mean(y):.2f}'
                      for x, y in run_helper.params.running_scales.items()]
            logger.info(
                f'Epoch: {epoch:3d}. '
                f'Batch: {i:5d}/{len(run_helper.task.train_loader)}. '
                f' Losses: {losses}.'
                f' Scales: {scales}')
            run_helper.params.running_losses = defaultdict(list)
            run_helper.params.running_scales = defaultdict(list)

    return


def test(run_helper: Helper, epoch, backdoor=False):
    model = run_helper.task.model
    model.eval()
    batch_acc = list()

    with torch.no_grad():
        for i, data in enumerate(run_helper.task.test_loader):
            batch = run_helper.task.get_batch(i, data)
            if backdoor:
                batch = run_helper.attack.backdoor.attack_batch(batch)

            outputs = model(batch.inputs)
            batch_acc.append(run_helper.task.get_batch_accuracy(outputs,
                                                                batch.labels))
    accuracy = np.mean(batch_acc)
    logger.info(f'Epoch: {epoch:4d} (Backdoor: {backdoor}). '
                f'Accuracy: {accuracy:.2f}')
    run_helper.plot(x=epoch, y=accuracy, name=f'accuracy/backdoor_{backdoor}')
    return np.mean(batch_acc)


def run(run_helper):
    for epoch in range(run_helper.params.start_epoch,
                       run_helper.params.epochs + 1):
        train(run_helper, epoch)
        acc = test(run_helper, epoch, backdoor=False)
        test(run_helper, epoch, backdoor=True)
        run_helper.save_model(run_helper.task.model, epoch, acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--commit', dest='commit', required=True)

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['commit'] = args.commit
    params['name'] = args.name

    helper = Helper(params)

    try:
        run(helper)
    except KeyboardInterrupt as e:
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
                if helper.params.tb:
                    shutil.rmtree(f'runs/{args.name}')
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. "
                             f"TB graph: {args.name}")
        else:
            logger.error(f"Aborted training. No output generated.")
