

import logging
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import CIFAR100
import numpy as np

from datasets.seq_cifar100 import TCIFAR100, MyCIFAR100
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_blurry_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args
from utils.prompt_templates import templates
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset


class SequentialCIFAR100224BLURRY30(ContinualDataset):
    """
    Sequential CIFAR-100 (224x224) with blurry class-incremental setting.
    Fully compatible with the new Mammoth data-loading pipeline.
    """

    NAME = 'seq-cifar100-224-blurry30'
    SETTING = 'class-il'
    N_TASKS = 10
    N_CLASSES = 100
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    SIZE = (224, 224)
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    BLURRY_M = 30  # percentage of blurry samples per task

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    def __init__(self, args, transform_type: str = 'weak'):
        super().__init__(args)

        assert transform_type in ['weak', 'strong']

        if transform_type == 'strong':
            logging.info('Using strong augmentation for CIFAR100-224')
            self.TRANSFORM = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD)
            ])

        # --------------------------------------------------
        # Load datasets (DO NOT wrap here)
        # --------------------------------------------------
        self.train_dataset = MyCIFAR100(
            base_path() + 'CIFAR100', train=True, download=True, transform=self.TRANSFORM
        )
        self.test_dataset = TCIFAR100(
            base_path() + 'CIFAR100', train=False, download=True, transform=self.TEST_TRANSFORM
        )

        # --------------------------------------------------
        # Build blurry indexes (PERMUTE-AWARE)
        # --------------------------------------------------
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

    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader, test_loader = store_blurry_masked_loaders(
            deepcopy(self.train_dataset),
            deepcopy(self.test_dataset),
            self,
            current_task=self.t
        )
        self.t += 1
        return train_loader, test_loader

    # --------------------------------------------------
    # Required API
    # --------------------------------------------------
    @staticmethod
    def get_transform():
        return transforms.Compose([
            transforms.ToPILImage(), SequentialCIFAR100224BLURRY30.TRANSFORM
        ])

    @set_default_from_args('backbone')
    def get_backbone(self):
        return 'vit'

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCIFAR100224BLURRY30.MEAN,
                                    SequentialCIFAR100224BLURRY30.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialCIFAR100224BLURRY30.MEAN,
                           SequentialCIFAR100224BLURRY30.STD)

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 20

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = CIFAR100(base_path() + 'CIFAR100', train=True,
                           download=True).classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates['cifar100']
