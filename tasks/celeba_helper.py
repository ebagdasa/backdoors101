from data_helpers.task_helper import TaskHelper


class CelebaHelper(TaskHelper):

    def load_data(self):
        logger.error('Celeba dataset is unfinished, needs more work')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            # transforms.RandomResizedCrop(178, scale=(0.9,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            # transforms.CenterCrop((178, 178)),
            # transforms.Resize((128, 128)),
            transforms.ToTensor(),
            normalize,
        ])

        self.train_dataset = CelebA(root=self.params.data_path,
                                    target_type='identity',  # ['identity',
                                    # 'attr'],
                                    split='train', transform=train_transform)

        self.test_dataset = CelebA(root=self.params.data_path,
                                   target_type='identity',
                                   split='test', transform=test_transform)

        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True, num_workers=8,
                                                  pin_memory=True)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False, num_workers=2)