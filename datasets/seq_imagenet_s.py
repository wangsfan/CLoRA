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

import torch
from utils import smart_joint

from datasets.utils import set_default_from_args
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode # type: ignore[import-untyped]


class MyImagenetS(Dataset):
    """
    Overrides dataset to change the getitem function.
    """
    IMG_SIZE = 224
    N_CLASSES = 1000

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True) -> None:
        self.not_aug_transform = transforms.Compose([
            transforms.Resize(MyImagenetS.IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        # if download:
        #     if os.path.isdir(root) and len(os.listdir(root)) > 0:
        #         logging.info('Download not needed, files already on disk.')
        #     else:
        #         from onedrivedownloader import download
        #         ln = '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21110&authkey=AIEfi5nlRyY1yaE" width="98" height="120" frameborder="0" scrolling="no"></iframe>'
        #         logging.info('Downloading dataset')
        #         download(ln, filename=smart_joint(root, 'cub_200_2011.zip'), unzip=True, unzip_path=root, clean=True)

        data_file = np.load(smart_joint(root, 'train_data.npz' if train else 'test_data.npz'), allow_pickle=True)

        self.data = data_file['data']
        self.targets = torch.from_numpy(data_file['targets']).long()
        self.classes = data_file['classes']

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
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

    def __len__(self) -> int:
        return len(self.data)


class ImagenetS(MyImagenetS):
    """Base CUB200 dataset."""

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False) -> None:
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

    def __getitem__(self, index: int, ret_segmask=False) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class SequentialImagenetS(ContinualDataset):
    NAME = 'seq-imagenet-s'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 50
    N_TASKS = 20
    N_CLASSES = 1000
    SIZE = (MyImagenetS.IMG_SIZE, MyImagenetS.IMG_SIZE)
    MEAN, STD = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(SIZE[0], interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    TEST_TRANSFORM = transforms.Compose([transforms.Resize(size=(224, 224),
                                                           interpolation=InterpolationMode.BICUBIC),
                                         transforms.CenterCrop(SIZE[0]),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=MEAN, std=STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = MyImagenetS(base_path() + 'sketch', train=True,
                                 download=True, transform=SequentialImagenetS.TRANSFORM)
        test_dataset = MyImagenetS(base_path() + 'sketch', train=False,
                              download=True, transform=SequentialImagenetS.TEST_TRANSFORM)

        train, test = store_masked_loaders(
            train_dataset, test_dataset, self)

        return train, test

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        pwd = os.path.dirname(os.path.abspath(__file__))
        with open(pwd + '/imagenet_s_utils/label_to_class_name.pkl', 'rb') as f:
            label_to_class_name = pickle.load(f)
        class_names = label_to_class_name.values()
        class_names = [x.replace('_', ' ') for x in class_names]

        class_names = fix_class_names_order(class_names, self.args)
        self.class_names = class_names
        return self.class_names

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialImagenetS.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone(self):
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialImagenetS.MEAN, std=SequentialImagenetS.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialImagenetS.MEAN, SequentialImagenetS.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128
