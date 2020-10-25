import torch
from backdoors.pattern_backdoor import PatternBackdoor


class SinglePixelBackdoor(PatternBackdoor):
    pattern_tensor = torch.tensor([[1]])