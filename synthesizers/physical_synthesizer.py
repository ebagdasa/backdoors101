import torch
from synthesizers.synthesizer import Synthesizer


class PhysicalSynthesizer(Synthesizer):
    """
    For physical backdoors it's ok to train using pixel pattern that
    represents the physical object in the real scene.
    """

    pattern_tensor = torch.tensor([[1.]])