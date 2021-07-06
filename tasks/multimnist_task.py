from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from dataset.multi_mnist_loader import MNIST
from tasks.mnist_task import MNISTTask


class MultiMNISTTask(MNISTTask):

    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        self.normalize])
        self.train_dataset = MNIST(root=self.params.data_path, train=True, download=True,
                                   transform=transform,
                                   multi=True)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True,
                                       num_workers=4)
        self.test_dataset = MNIST(root=self.params.data_path, train=False, download=True,
                                  transform=transform,
                                  multi=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size, shuffle=False,
                                      num_workers=4)
        self.classes = list(range(100))
