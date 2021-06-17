import torch

from synthesizers.pattern_synthesizer import PatternSynthesizer


class TimesSynthesizer(PatternSynthesizer):
    """
    Synthesizer of a 'times' ('X') sign pattern, multiplying numerical labels results
    """

    pattern_tensor = torch.tensor([
        [1., -10., 1.],
        [-10., 1., -10.],
        [1., -10., 1.],
    ])

    def synthesize_labels(self, batch, attack_portion=None):
        batch.labels[:attack_portion] = (batch.labels[:attack_portion] / 10).int() * (batch.labels[:attack_portion] % 10).int()
        return
