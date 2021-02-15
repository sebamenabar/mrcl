import logging
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToTensor

import datasets.omniglot as om

# import numpy as np

# train_train_mean = (0.9195019446031442, 0.9195019446031442, 0.9195019446031442)
# train_train_std = (0.23467870686722328, 0.23467870686722328, 0.23467870686722328)
train_train_mean = 0.9195019446031442
train_train_std = 0.23467870686722328


logger = logging.getLogger("experiment")


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(
        name,
        train=True,
        path=None,
        background=True,
        all=False,
        prefetch_gpu=False,
        device=None,
        resize=None,
        augment=False,
        normalize=False,
    ):

        if name == "omniglot":
            if resize is None:
                logger.info("Using image size 84")
                resize = 84
            else:
                logger.info(f"Using image size {resize}")

            if augment:
                logger.info("Using data augmentation")
                train_transform = transforms.Compose(
                    [
                        transforms.Resize(resize),
                        transforms.RandomCrop(resize, padding=8),
                        transforms.ToTensor(),
                        transforms.Normalize(train_train_mean, train_train_std),
                    ]
                )
                # warmup_transform = transforms.Compose(
                #     [
                #         transforms.Resize(resize),
                #         transforms.ToTensor(),
                #         transforms.Normalize(train_train_mean, train_train_std),
                #     ]
                # )
                # warmup_steps = 0
            else:
                logger.info("NO augmentation")
                train_transform = transforms.Compose(
                    [
                        transforms.Resize(resize),
                        transforms.ToTensor(),
                        transforms.Normalize(train_train_mean, train_train_std),
                    ]
                )
            warmup_transform = None
            warmup_steps = 0
            if path is None:
                return om.Omniglot(
                    "../data/omni",
                    background=background,
                    download=True,
                    train=train,
                    transform=train_transform,
                    all=all,
                    prefetch_gpu=prefetch_gpu,
                    device=device,
                    warmup_transform=warmup_transform,
                    warmup_steps=warmup_steps,
                )
            else:
                return om.Omniglot(
                    path,
                    background=background,
                    download=True,
                    train=train,
                    transform=train_transform,
                    all=all,
                    prefetch_gpu=prefetch_gpu,
                    device=device,
                    warmup_transform=warmup_transform,
                    warmup_steps=warmup_steps,
                )

        else:
            print("Unsupported Dataset")
            assert False
