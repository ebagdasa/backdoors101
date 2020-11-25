from tasks.fl.fl_task import FederatedLearningTask


class RedditTask(FederatedLearningTask):

    def load_data(self) -> None:
        return