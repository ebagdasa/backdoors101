import random
from utils.helper import Helper
import torch.nn as nn
import torch.optim as optim


class Trainer:

    def __init__(self, params, model: nn.Module, dataset):
        self.params = params
        self.model = model
        self.dataset = dataset
        self.optimizer_type = self.params.get('optimizer_type', None)
        self.lr = self.params.get('lr', 0)
        self.decay = self.params.get('decay', 0.0005)
        self.momentum = self.params.get('momentum', 0.9)

        self.optimizer = self.make_optimizer()
        self.optimizer = self.make_optimizer()

    def make_optimizer(self):
        if self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                                       weight_decay=self.decay, momentum=self.momentum)
        elif self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f'No optimizer: {self.optimizer}')

        return

    def train(self):

        self.model.train()

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

    def test(self, model: nn.Module, criterion, epoch, is_poison=False):
        model.eval()
        total_loss = 0.0
        correct = 0.0
        total_test_words = 0.0
        hidden = model.init_hidden(run_helper.params['test_batch_size'])
        data_source = run_helper.test_data
        data_iterator = range(0, data_source.size(0) - 1, run_helper.params['bptt'])
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
            total_l = total_loss.item() / (dataset_size - 1)
            logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                           total_l, correct, total_test_words,
                                                           acc))
            acc = acc.item()

        return acc, total_l
