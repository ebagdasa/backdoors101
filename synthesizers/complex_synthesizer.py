import torch

from synthesizers.pattern_synthesizer import PatternSynthesizer


class ComplexSynthesizer(PatternSynthesizer):
    """
    For physical backdoors it's ok to train using pixel pattern that
    represents the physical object in the real scene.
    """

    pattern_tensor = torch.tensor([[1.]])

    def synthesize_labels(self, batch, attack_portion=None):
        batch.labels[:attack_portion] = batch.aux[:attack_portion]
        return
