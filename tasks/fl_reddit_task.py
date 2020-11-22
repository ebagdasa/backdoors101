from tasks.fl_task import FederatedLearningTask


class RedditTask(FederatedLearningTask):

    def load_data(self) -> None:
        return