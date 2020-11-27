import torch
from metrics.metric import Metric


class TestLossMetric(Metric):

    def __init__(self, criterion, train=False):
        self.criterion  = criterion
        self.main_metric_name = 'value'
        super().__init__(name='Loss', train=False)

    def compute_metric(self, outputs: torch.Tensor,
                       labels: torch.Tensor, top_k=(1,)):
        """Computes the precision@k for the specified values of k"""
        loss = self.criterion(outputs, labels)
        return {'value': loss.mean().item()}