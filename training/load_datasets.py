import wilds
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import random_split, Dataset, Subset

import sys, os

sys.path.append(os.path.dirname(__file__))
from corruptions import validation_corruptions, test_corruptions

# Function to standarize a tensor (0 mean and 1 std)
def standardize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(1, 2))
    std = x.std(dim=(1, 2))
    std[std == 0.0] = 1.0
    return TF.normalize(x, mean, std)


# Setup the default transformations for wilds datasets
default_transforms = {
    "camelyon17": [
        transforms.Resize(size=(96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    "iwildcam": [
        transforms.Resize(size=(96, 96)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: standardize(x)),
    ],
    "rxrx1": [
        transforms.Resize(size=(96, 96)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: standardize(x)),
    ],
}


def load_dataset(data_dir, name):
    if name == "camelyon17-train":
        return load_wilds_dataset("camelyon17", data_dir, split="train")
    elif name == "camelyon17-id_val":
        id_val_dataset = load_wilds_dataset("camelyon17", data_dir, split="id_val")
        val, test = random_split(
            id_val_dataset, [0.5, 0.5], generator=torch.Generator().manual_seed(0)
        )
        return val
    elif name == "camelyon17-c1-val":
        return load_wilds_dataset(
            "camelyon17",
            data_dir,
            split="id_val",
            corruptions=[transforms.RandomChoice(validation_corruptions(1))],
        )
    elif name == "camelyon17-c1-test":
        return load_wilds_dataset(
            "camelyon17",
            data_dir,
            split="id_val",
            corruptions=[transforms.RandomChoice(test_corruptions(1))],
        )
    elif name == "camelyon17-val":
        return load_wilds_dataset("camelyon17", data_dir, split="val")
    elif name == "camelyon17-id_test":
        id_val_dataset = load_wilds_dataset("camelyon17", data_dir, split="id_val")
        val, test = random_split(
            id_val_dataset, [0.5, 0.5], generator=torch.Generator().manual_seed(0)
        )
        return test
    elif name == "camelyon17-test":
        return load_wilds_dataset("camelyon17", data_dir, split="test")
    elif name == "rxrx1-id_test":
        return load_wilds_dataset("rxrx1", data_dir, split="id_test")
    elif name == "gaussian_noise-(96x96)":
        return random_noise_dataset(items=10000, size=(96, 96), channels=3)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def random_noise_dataset(items=1000, size=(96, 96), channels=3):
    Xtrain = torch.randn(items, channels, *size)
    ytrain = torch.randn(items, 1)
    md = torch.randn(items, 1)
    return torch.utils.data.TensorDataset(Xtrain, ytrain, md)


def load_wilds_dataset(
    dataset_name,
    dir,
    split="test",
    corruptions=[],
):
    def_transforms = default_transforms[dataset_name]
    dataset = wilds.get_dataset(dataset=dataset_name, root_dir=dir)
    transform = transforms.Compose([*corruptions, *def_transforms])
    return dataset.get_subset(split, transform=transform)
