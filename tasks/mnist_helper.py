from data_helpers.task_helper import TaskHelper


class MNISTHelper(TaskHelper):


    def load_data(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.train_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)
        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=2)
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 num_workers=2)
        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        return True

    def iter_batches(self, train=True):
        data_loader = self.train_loader if train else self.test_loader
        for input, label in data_loader:
            yield (input, label)