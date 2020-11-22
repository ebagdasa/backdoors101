import argparse
import shutil
from datetime import datetime

import numpy as np
import yaml
from prompt_toolkit import prompt

# noinspection PyUnresolvedReferences
from dataset.pipa import Annotations  # legacy to correctly load dataset.
from helper import Helper
from tasks.fl_task import FederatedLearningTask
from utils.utils import *

logger = logging.getLogger('logger')


def train(hlpr: Helper, epoch, model, optimizer, train_loader):
    criterion = hlpr.task.criterion
    model.train()

    for i, data in enumerate(train_loader):
        if i == hlpr.params.max_batch_id:
            break
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()

        loss = hlpr.attack.compute_blind_loss(model, criterion, batch)

        loss.backward()

        optimizer.step()

        hlpr.report_training_losses_scales(i, epoch)

    return


def test(hlpr: Helper, epoch, backdoor=False):
    model = hlpr.task.model
    model.eval()
    batch_acc = list()

    with torch.no_grad():
        for i, data in enumerate(hlpr.task.test_loader):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.attack_batch(batch,
                                                             test=True)

            outputs = model(batch.inputs)
            batch_acc.append(hlpr.task.get_batch_accuracy(outputs,
                                                          batch.labels))
    accuracy = np.mean(batch_acc)
    logger.info(f'Epoch: {epoch:4d} (Backdoor: {backdoor}). '
                f'Accuracy: {accuracy:.2f}')
    hlpr.plot(x=epoch, y=accuracy, name=f'Accuracy/backdoor_{backdoor}')
    return np.mean(batch_acc)


def run(hlpr):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        acc = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)
        hlpr.save_model(hlpr.task.model, epoch, acc)


def fl_run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        run_fl_round(hlpr)


def run_fl_round(hlpr):
    global_model = hlpr.task.model
    sampled_users = hlpr.task.sample_users_for_round()
    weight_accumulator = hlpr.task.get_empty_accumulator()

    for user_id, train_loader in sampled_users:
        local_model, optimizer = hlpr.task.get_local_model_optimizer()
        for local_epoch in range(hlpr.params.fl_local_epochs):
            train(hlpr, local_epoch, local_model, optimizer, train_loader)
            local_update = hlpr.task.get_fl_update(local_model, global_model)

            hlpr.task.accumulate_weights(weight_accumulator, local_update)

    hlpr.task.update_global_model(weight_accumulator, global_model)


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
        if issubclass(FederatedLearningTask, helper.task.__class__):
            fl_run(helper)
        else:
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
