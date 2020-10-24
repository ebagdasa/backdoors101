from data_helpers.task_helper import TaskHelper


class MultiMNISTHelper(TaskHelper):


    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,),
                                                             (0.3081,))])
        self.train_dataset = MNIST(root='./data', train=True, download=True,
                                   transform=transform,
                                   multi=True)
        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=4)
        self.test_dataset = MNIST(root='./data', train=False, download=True,
                                  transform=transform,
                                  multi=True)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=100, shuffle=True,
                                                 num_workers=4)
        self.classes = list(range(100))