import torch

from backdoors.backdoor import Backdoor
from tasks.batch import Batch
from tasks.task import Task


class PatternBackdoor(Backdoor):

    pattern_tensor: torch.Tensor = torch.tensor([
        [1, 0, 1],
        [-10, 1, -10],
        [-10, -10, 0],
        [-10, 1, -10],
        [1, 0, 1]
    ])
    "Just some random 2D pattern."

    x_top = 3
    "X coordinate to put the backdoor into."
    y_top = 23
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def __init__(self, task: Task):
        super().__init__(task)
        self.make_pattern(self.pattern_tensor, self.x_top, self.y_top)

    def make_pattern(self, pattern_tensor, x_top, y_top):
        full_image = torch.zeros(self.task.input_shape)
        full_image.fill_(self.mask_value)

        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]

        if x_bot >= self.task.input_shape[1] or \
                y_bot >= self.task.input_shape[2]:
            raise ValueError(f'Position of backdoor outside image limits:'
                             f'image: {self.task.input_shape}, but backdoor'
                             f'ends at ({x_bot}, {y_bot})')

        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

        self.mask = 1 * (full_image != self.mask_value).to(self.params.device)
        self.pattern = self.task.normalize(full_image).to(self.params.device)


    def apply_backdoor(self, batch: Batch, attack_proportion):
        inputs = (1 - self.mask) * batch.inputs[:attack_proportion] \
                 + self.mask * self.pattern
        batch.inputs[:attack_proportion] = inputs
        batch.labels[:attack_proportion].fill_(self.params.backdoor_label)
        return batch
