from data.dataset import Dataset


class Task:

    optimizer = None
    scheduler = None
    model_params = None

    def __init__(self, params, dataset: Dataset, ):

        self.params = params
        self.lr = self.params.get('lr', None)
        self.optimizer = self.params.get('optimizer', None)
        self.decay = self.params.get('decay', None)
        self.momentum = self.params.get('momentum', None)
        self.epochs = self.params.get('epochs', None)
        self.dataset = get_dataset()
        self.model = get_model(params['model_params'])

    def get_data(self):
        raise NotImplemented

    def next_batch(self):
        raise NotImplemented

    def forward(self):
        raise NotImplemented

