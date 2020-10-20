from data.dataset import Dataset


class Task:

    optimizer = None
    scheduler = None
    model_params = None

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        return

    def get_data(self):
        raise NotImplemented

    def next_batch(self):
        raise NotImplemented

    def make_model(self):
        raise NotImplemented

    def forward(self):
        raise NotImplemented




