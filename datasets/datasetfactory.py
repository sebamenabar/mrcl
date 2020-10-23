import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToTensor

import datasets.omniglot as om


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
    ):

        if name == "omniglot":
            if resize is None:
                print("Using image size 84")
                resize = 84
            else:
                print(f"Using image size {resize}")

            if augment:
                print("Using data augmentation")
                train_transform = transforms.Compose(
                    [
                        transforms.Resize(resize),
                        transforms.RandomCrop(resize, padding=8),
                        transforms.ToTensor(),
                    ]
                )
                warmup_transform = transforms.Compose(
                    [transforms.Resize(resize), transforms.ToTensor()]
                )
                warmup_steps = 0
            else:
                print("NO augmentation")
                train_transform = transforms.Compose(
                    [transforms.Resize(resize), transforms.ToTensor()]
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
