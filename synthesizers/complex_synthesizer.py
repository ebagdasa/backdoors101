import torch

from synthesizers.pattern_synthesizer import PatternSynthesizer


class ComplexSynthesizer(PatternSynthesizer):
    """
    For physical backdoors it's ok to train using pixel pattern that
    represents the physical object in the real scene.
    """

    pattern_tensor = torch.tensor([[1.]])


    def synthesize_labels(self, batch):
        batch.labels = batch.aux
        return