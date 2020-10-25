from torchvision.transforms import transforms
import torch
import torch.utils.data as torch_data

from dataset.pipa import *
from tasks.task import Task


class PipaHelper(Task):

    def load_data(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.train_dataset = PipaDataset(train=True, transform=train_transform)
        # poison_weights = [0.99, 0.004, 0.004, 0.004, 0.004]
        # weights = [14081, 4893, 1779, 809,
        #            862]  # [0.62, 0.22, 0.08, 0.04, 0.04]
        weights = torch.tensor([0.03, 0.07, 0.2, 0.35, 0.35])
        weights_labels = weights[self.train_dataset.labels]
        sampler = torch_data.sampler.WeightedRandomSampler(weights_labels, len(
            self.train_dataset))
        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  sampler=sampler)

        self.test_dataset = PipaDataset(train=False, transform=test_transform)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False, num_workers=2)

        self.classes = list(range(5))


    def iter_batches(self, train=True):
        data_loader = self.train_loader if train else self.test_loader
        for input, label, second_label, _ in data_loader:
            yield Batch(input, label, second_label)
