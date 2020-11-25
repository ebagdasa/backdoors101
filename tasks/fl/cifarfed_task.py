import random
from collections import defaultdict

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from tasks.cifar10_task import Cifar10Task
from tasks.fl.fl_task import FederatedLearningTask


class CifarFedTask(FederatedLearningTask, Cifar10Task):

    def load_data(self) -> None:
        self.load_cifar_data()
        if self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
        else:
            # sample indices for participants that are equally
            # split to 500 images per participant
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [self.get_train_old(all_range, pos)
                             for pos in
                             range(self.params.fl_total_participants)]
        self.fl_train_loaders = train_loaders
        return

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indices dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as
            parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params.poison_images or \
                    ind in self.params.poison_images_test:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][
                               :min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][
                                   min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param indices:
        :return:
        """
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.params.batch_size,
                                  sampler=SubsetRandomSampler(
                                      indices))
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(
            len(self.train_dataset) / self.params.fl_total_participants)
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.params.batch_size,
                                  sampler=SubsetRandomSampler(
                                      sub_indices))
        return train_loader
