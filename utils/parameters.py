from dataclasses import dataclass
from typing import List
import logging
import os
logger = logging.getLogger('logger')


@dataclass
class Params:

    current_time: str = None
    name: str = None
    commit: float = None
    random_seed: int = None

    # training params
    device: str = 'cuda'
    start_epoch: float = None
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
    batch_size: int = 64
    test_batch_size: int = 100

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
    losses: List[str] = None
    normalize: float = None
    # relabel images with poison_number
    poison_images: float = None
    poison_images_test: float = None
    # optimizations:
    alternating_attack: float = None
    clip_batch: float = None
    # Disable BatchNorm and Dropout
    switch_to_eval: float = None

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
        self.folder_path = f'saved_models/model_{self.name}_{self.dataset}_{self.current_time}'

        # enabling logging anyways when saving models
        if self.save_model or self.tb:
            self.log = True

        if self.log:
            try:
                os.mkdir(self.folder_path)
            except FileExistsError:
                logger.info('Folder already exists')

            # add a line to html file with links (useful for quick navigation)
            with open('saved_models/runs.html', 'a') as f:
                f.writelines([f'<div><a href="https://github.com/ebagdasa/backdoors/tree/{self.commit}">GitHub</a>,'
                              f'<span> <a href="http://gpu/{self.folder_path}">{self.name}_'
                              f'{self.current_time}</a></div>'])
        else:
            self.folder_path = None
