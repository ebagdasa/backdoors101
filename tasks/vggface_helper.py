from data_helpers.task_helper import TaskHelper


class VggFaceHelper(TaskHelper):

    def load_data(self):
        logger.error('VGG dataset is unfinished, needs more work')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.train_dataset = VGG_Faces2(
            root=self.params.data_path,
            train=True, transform=transform_train)
        self.test_dataset = VGG_Faces2(
            root=self.params.data_path,
            train=False, transform=transform_test)

        return True

