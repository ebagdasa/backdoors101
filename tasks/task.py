

class Task:

    optimizer = None
    scheduler = None
    model_params = None

    def __init__(self):
        return

    def get_data(self):
        raise NotImplemented

    def next_batch(self):
        raise NotImplemented

    def make_model(self):
        raise NotImplemented




