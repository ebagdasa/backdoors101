from dataclasses import dataclass
import torch


@dataclass
class Batch:
    batch_id: int
    inputs: torch.Tensor
    labels: torch.Tensor

    # For PIPA experiment we use this field to store identity label.
    aux: torch.Tensor = None

    def __post_init__(self):
        self.batch_size = self.inputs.shape[0]

    def to(self, device):
        inputs = self.inputs.to(device)
        labels = self.labels.to(device)
        if self.aux is not None:
            aux = self.aux.to(device)
        else:
            aux = None
        return Batch(self.batch_id, inputs, labels, aux)

    def clone(self):
        inputs = self.inputs.clone()
        labels = self.labels.clone()
        if self.aux is not None:
            aux = self.aux.clone()
        else:
            aux = None
        return Batch(self.batch_id, inputs, labels, aux)


    def clip(self, batch_size):
        if batch_size is None:
            return self

        inputs = self.inputs[:batch_size]
        labels = self.labels[:batch_size]

        if self.aux is None:
            aux = None
        else:
            aux = self.aux[:batch_size]

        return Batch(self.batch_id, inputs, labels, aux)