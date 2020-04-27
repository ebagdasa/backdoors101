#!/usr/bin/env python

import collections
import os

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import torchvision.transforms
from torchvision.datasets.folder import default_loader


class VGG_Faces2(data.Dataset):

    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt

    def __init__(self, root, train, transform=None):
        """
        :param root: dataset directory
        :param image_list_file: contains image file names under root
        :param id_label_dict: X[class_id] -> label
        :param split: train or valid
        :param transform:
        :param horizontal_flip:
        :param upper: max number of image used for debug
        """
        self.root = root

        if train:
            self.file_list = torch.load(self.root + '/train_list.pt')
        else:
            self.file_list = torch.load(self.root + '/test_list.pt')
        self.bboxes = torch.load(self.root + '/bboxes.pt')

        self.transform = transform
        self.loader = default_loader

        # self.img_info = []
        # with open(self.image_list_file, 'r') as f:
        #     for i, img_file in enumerate(f):
        #         img_file = img_file.strip()  # e.g. train/n004332/0317_01.jpg
        #         class_id = img_file.split("/")[1]  # like n004332
        #         label = self.id_label_dict[class_id]
        #         self.img_info.append({
        #             'cid': class_id,
        #             'img': img_file,
        #             'lbl': label,
        #         })
        #         if i % 1000 == 0:
        #             print("processing: {} images for {}".format(i, self.split))
        #         if upper and i == upper - 1:  # for debug purpose
        #             break

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_file, label, bbox_id = self.file_list[index]
        bbox = self.bboxes[bbox_id]
        sample = self.loader(f'{self.root}/{img_file}')
        target = torch.tensor(label)
        x, y, w, h = bbox

        sample = sample.crop((x,y, x+w, y+h))

        if self.transform:
            sample = self.transform(sample)

        return sample, target

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl