import logging
try:
    import requests
except ImportError as e:
    logging.error("Please install requests using 'pip install requests'")
    raise e

import os
import torchvision.transforms as transforms # type: ignore[import-untyped]
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pickle
from PIL import Image
from typing import Tuple

import yaml

from datasets.utils import set_default_from_args
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_blurry_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode # type: ignore[import-untyped]
from copy import deepcopy

class MyImagenetR(Dataset):
    N_CLASSES = 200

    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.not_aug_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC), transforms.ToTensor()])

        if not os.path.exists(self.root):
            if download:
                os.makedirs(self.root, exist_ok=True)
                # download from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
                logging.info("Downloading imagenet-r dataset...")
                url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
                r = requests.get(url, allow_redirects=True)
                if not os.path.exists(self.root):
                    os.makedirs(self.root)
                logging.info("Writing tar on disk...")
                open(self.root + 'imagenet-r.tar', 'wb').write(r.content)
                logging.info("Extracting tar...")
                os.system('tar -xf ' + self.root + 'imagenet-r.tar -C ' + self.root.rstrip('imagenet-r'))

                # move all files in imagenet-r to root with shutil
                import shutil
                logging.info("Moving files...")
                for d in os.listdir(self.root + 'imagenet-r'):
                    shutil.move(self.root + 'imagenet-r/' + d, self.root)

                logging.info("Cleaning up...")
                os.remove(self.root + 'imagenet-r.tar')
                os.rmdir(self.root + 'imagenet-r')

                logging.info("Done!")
            else:
                raise RuntimeError('Dataset not found.')

        pwd = os.path.dirname(os.path.abspath(__file__))
        if self.train:
            data_config = yaml.load(open(pwd + '/imagenet_r_utils/imagenet-r_train.yaml'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open(pwd + '/imagenet_r_utils/imagenet-r_test.yaml'), Loader=yaml.Loader)

        self.data = np.array(data_config['data'])
        self.targets = np.array(data_config['targets'])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.open(img).convert('RGB')

        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialImagenetRBLURRY10(ContinualDataset):

    NAME = 'seq-imagenet-r-blurry10'
    SETTING = 'class-il'
    N_TASKS = 10
    N_CLASSES = 200
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    MEAN, STD = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    SIZE = (224, 224)
    BLURRY_M = 10

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(SIZE[0], interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    TEST_TRANSFORM = transforms.Compose([transforms.Resize(size=(256, 256),
                                                           interpolation=InterpolationMode.BICUBIC),
                                         transforms.CenterCrop(SIZE[0]),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=MEAN, std=STD)])

    def __init__(self, args):
        super().__init__(args)
        self.train_dataset = MyImagenetR(base_path() + 'imagenet-r/', train=True,
                                            download=True, transform=self.TRANSFORM)

        self.test_dataset = MyImagenetR(base_path() + 'imagenet-r/', train=False,
                                           download=True, transform=self.TEST_TRANSFORM)

        train_targets = np.array(self.train_dataset.targets)
        test_targets = np.array(self.test_dataset.targets)

        if args.permute_classes:
            train_targets = args.class_order[train_targets]
            test_targets = args.class_order[test_targets]

        self.train_idx_all = []
        self.test_idx_all = []
        self.blurry_idx_all = []

        for t in range(self.N_TASKS):
            cls_start = t * self.N_CLASSES_PER_TASK
            cls_end = (t + 1) * self.N_CLASSES_PER_TASK

            trn_idx = np.where((train_targets >= cls_start) & (train_targets < cls_end))[0]
            tst_idx = np.where((test_targets >= cls_start) & (test_targets < cls_end))[0]

            # blurry samples from current task
            n_blurry = int(len(trn_idx) * self.BLURRY_M / 100)
            blurry_idx = np.random.choice(trn_idx, n_blurry, replace=False)
            clean_idx = np.setdiff1d(trn_idx, blurry_idx)

            self.train_idx_all.append(clean_idx)
            self.test_idx_all.append(tst_idx)

            # split blurry samples across other tasks
            blurry_splits = np.array_split(blurry_idx, self.N_TASKS - 1)
            blurry_splits = list(blurry_splits)
            blurry_splits.insert(t, np.array([], dtype=np.int64))
            self.blurry_idx_all.append(blurry_splits)

        # inject blurry samples into other tasks
        for t in range(self.N_TASKS):
            other_tasks = np.delete(np.arange(self.N_TASKS), t)
            injected = np.concatenate([self.blurry_idx_all[ot][t] for ot in other_tasks])
            self.train_idx_all[t] = np.concatenate([self.train_idx_all[t], injected])

        # blurry task pointer (independent from c_task)
        self.t = 0

    def get_data_loaders(self):
        train_loader, test_loader = store_blurry_masked_loaders(
            deepcopy(self.train_dataset),
            deepcopy(self.test_dataset),
            self,
            current_task=self.t
        )
        self.t += 1
        return train_loader, test_loader

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        pwd = os.path.dirname(os.path.abspath(__file__))
        with open(pwd + '/imagenet_r_utils/label_to_class_name.pkl', 'rb') as f:
            label_to_class_name = pickle.load(f)
        class_names = label_to_class_name.values()
        class_names = [x.replace('_', ' ') for x in class_names]

        class_names = fix_class_names_order(class_names, self.args)
        self.class_names = class_names
        return self.class_names

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialImagenetRBLURRY10.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone(self):
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialImagenetRBLURRY10.MEAN, std=SequentialImagenetRBLURRY10.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialImagenetRBLURRY10.MEAN, SequentialImagenetRBLURRY10.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128
