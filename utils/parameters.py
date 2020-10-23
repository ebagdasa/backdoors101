from dataclasses import dataclass, asdict
from typing import List
import logging
import torch
logger = logging.getLogger('logger')

ALL_TASKS =  ['backdoor', 'normal', 'latent_fixed', 'latent', 'ewc',
                           'neural_cleanse', 'mask_norm', 'sums']

@dataclass
class Params:

    current_time: str = None
    name: str = None
    commit: float = None
    random_seed: int = None
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # training params
    start_epoch: int = 1
    epochs: int = None
    log_interval: int = 1000
    # model
    model_type: str = 'Simple'
    pretrained: bool = False
    resume_model: str = None
    lr: float = None
    decay: float = None
    momentum: float = None
    optimizer: str = None
    scheduler: str = None
    # data
    dataset: str = 'MNIST'
    data_path: str = '.data/'
    batch_size: int = 64
    test_batch_size: int = 100
    transform_train: bool = True

    # gradient shaping/DP params
    dp: bool = None
    dp_clip: float = None
    dp_sigma: float = None

    # attack params
    backdoor: bool = False
    backdoor_label: int = 8
    poisoning_proportion: float = 1.0  # backdoors proportion in backdoor loss
    pattern_type: str = 'pattern'
    # losses to balance: `normal`, `backdoor`, `neural_cleanse`, `sentinet`
    loss_tasks: List[str] = None
    normalize: float = None
    # relabel images with poison_number
    poison_images: List[int] = None
    poison_images_test: List[int] = None
    # optimizations:
    alternating_attack: float = None
    clip_batch: float = None
    # Disable BatchNorm and Dropout
    switch_to_eval: float = None

    # nc evasion
    nc_p_norm: int = 1

    # logging
    log: bool = False
    tb: bool = False
    save_model: bool = None
    save_on_epochs: List[int] = None
    save_scale_values: bool = False
    print_memory_consumption: bool = False
    save_timing: bool = False

    # future FL params
    alpha: float = None

    def __post_init__(self):
        # enable logging anyways when saving statistics
        if self.save_model or self.tb or self.save_timing or \
                self.print_memory_consumption:
            self.log = True

        if self.log:
            self.folder_path = f'saved_models/model_{self.name}_' \
                               f'{self.dataset}_{self.current_time}'

        for t in self.loss_tasks:
            if t not in ALL_TASKS:
                raise ValueError(f'Task {t} is not part of the supported '
                                 f'tasks: {ALL_TASKS}.')

    def to_dict(self):
        return asdict(self)