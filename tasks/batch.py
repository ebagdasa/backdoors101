from dataclasses import dataclass
import torch


@dataclass
class Batch:
    inputs: torch.Tensor
    labels: torch.Tensor

    # For PIPA experiment we use this field to store identity label.
    aux: torch.Tensor = None

    def to(self, device):
        self.inputs = self.inputs.to(device)
        self.labels = self.labels.to(device)
        if self.aux is not None:
            self.aux = self.aux.to(device)
