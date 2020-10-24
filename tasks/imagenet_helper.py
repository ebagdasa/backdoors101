import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import transforms

from data_helpers.task_helper import TaskHelper


class ImageNetHelper(TaskHelper):

    def load_data(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        self.train_dataset = torchvision.datasets.ImageNet(
            root=self.params.data_path,
            split='train', transform=train_transform)

        self.test_dataset = torchvision.datasets.ImageNet(
            root=self.params.data_path,
            split='val', transform=test_transform)

        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True, num_workers=2)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False, num_workers=2)

        with open(
                f'{self.params.data_path}/imagenet1000_clsidx_to_labels.txt') \
                as f:
            self.classes = eval(f.read())
