import wilds
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader


def random_noise_dataset(items=1000, size=(96,96), channels=3, batch_size=32, pin_memory=True):
    Xtrain = torch.randn(items, channels, *size)
    ytrain = torch.randn(items, 1)
    md = torch.randn(items, 1)
    dataset = torch.utils.data.TensorDataset(Xtrain, ytrain, md)
    return DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=pin_memory)


def standardize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(1, 2))
    std = x.std(dim=(1, 2))
    std[std == 0.] = 1.
    return TF.normalize(x, mean, std)


def load_wilds_dataset(dataset_name, 
                       dir,
                       split="test",
                       shuffle=False,
                       batch_size=32,
                       pin_memory=True, 
                       resize=(96,96),
                       normalization_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])):
    dataset = wilds.get_dataset(dataset=dataset_name, root_dir=dir)
    
    transform = transforms.Compose(
        [transforms.Resize(size=resize),
        transforms.ToTensor(),
        normalization_transform if  dataset_name in ["camelyon17"] else transforms.Lambda(lambda x: standardize(x))
        ]
    )
    datasubset = dataset.get_subset(split, transform=transform)
    return DataLoader(datasubset, shuffle=shuffle, batch_size=batch_size, pin_memory=pin_memory)