from __future__ import print_function, division

import torch
import torch.utils.data as data
from torchvision.datasets.folder import default_loader


class Annotations:
    photoset_id = None
    photo_id = None
    xmin = None
    ymin = None
    width = None
    height = None
    identity_id = None
    subset_id = None
    people_on_photo = 0

    def __repr__(self):
        return f'photoset: {self.photoset_id}, photo id: {self.photo_id}, ' \
               f'identity: {self.identity_id}, subs: {self.subset_id}, ' \
               f'{self.people_on_photo}'


class PipaDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, train=True, transform=None):
        """
        Args:
            data_path (string): Directory with all the data.
            train (bool): train or test dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.directory = data_path
        try:
            if train:
                self.data_list = torch.load(f'{self.directory}/train_split.pt')
            else:
                self.data_list = torch.load(f'{self.directory}/test_split.pt')
            self.photo_list = torch.load(f'{self.directory}/photo_list.pt')
            self.target_identities = torch.load(
                f'{self.directory}/target_identities.pt')
        except FileNotFoundError:
            raise FileNotFoundError(
                'Please download the archive: https://drive.google.com/'
                'file/d/1IAsTDl6kw4u8kk7Ikyf8K2A4RSPv9izz')
        self.transform = transform
        self.loader = default_loader

        self.labels = torch.tensor(
            [self.get_label(x)[0] for x in range(len(self))])
        self.metadata = [self.get_label(x) for x in range(len(self))]

    def __len__(self):
        return len(self.data_list)

    def get_label(self, idx):
        photo_id, identities = self.data_list[idx]
        target = len(identities) - 1
        if target > 4:
            target = 4
        target_identity = 0
        for pos, z in enumerate(self.target_identities):
            if z in identities:
                target_identity = pos + 1
        return target, target_identity, photo_id, idx

    def __getitem__(self, idx):
        photo_id, identities = self.data_list[idx]
        x = self.photo_list[photo_id][0]
        if x.subset_id == 1:
            path = 'train'
        else:
            path = 'test'

        target = len(identities) - 1

        # more than 5 people nobody cares
        if target > 4:
            target = 4
        target_identity = 0
        for pos, z in enumerate(self.target_identities):
            if z in identities:
                target_identity = pos + 1

        # get image
        sample = self.loader(
            f'{self.directory}/{path}/{x.photoset_id}_{x.photo_id}.jpg')
        crop = self.get_crop(photo_id)
        sample = sample.crop(crop)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, target_identity, (photo_id, idx)

    def get_crop(self, photo_id):
        ids = self.photo_list[photo_id]

        left = 100000
        upper = 100000
        right = 0
        lower = 0
        for x in ids:
            left = min(x.xmin, left)
            upper = min(x.ymin, upper)
            right = max(x.xmin + x.width, right)
            lower = max(x.ymin + x.height, lower)

        diff = (right - left) - (lower - upper)
        if diff >= 0:
            lower += diff
        else:
            right -= diff

        return left, upper, right, lower
