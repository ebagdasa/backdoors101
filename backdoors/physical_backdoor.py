import random

from torchvision.transforms import transforms

from backdoors.backdoor import Backdoor


class PhysicalBackdoor(Backdoor):

    def apply_backdoor(self, batch, i=-1):
        resize = random.randint(6, 16)
        pattern = get_pattern(helper.pattern)
        if random.random() > 0.5:
            pattern = transforms.functional.hflip(pattern)
        pattern = trans_tens(transforms.functional.resize(pattern, resize,
                                                          interpolation=0)).squeeze()
        pattern *= max_val
        pattern[pattern == 0] += min_val
        x = pattern.shape[-2]
        y = pattern.shape[-1]
        raise NotImplemented