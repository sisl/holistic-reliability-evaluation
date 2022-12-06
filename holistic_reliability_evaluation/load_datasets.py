import wilds
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# This comes from imagenet_cbar
import converters
from transform_finder import transform_dict, build_transform


def random_noise_dataset(items=1000, size=(96,96), channels=3, batch_size=32, pin_memory=True):
    Xtrain = torch.randn(items, channels, *size)
    ytrain = torch.randn(items, 1)
    md = torch.randn(items, 1)
    return torch.utils.data.TensorDataset(Xtrain, ytrain, md) 


def standardize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(1, 2))
    std = x.std(dim=(1, 2))
    std[std == 0.] = 1.
    return TF.normalize(x, mean, std)

def corruption_transforms(severity):
    return [build_transform(name, severity, 'imagenet')  for name in set(transform_dict.keys()).difference({'color_balance'})]


def load_wilds_dataset(dataset_name, 
                       dir,
                       split="test",
                       resize=(96,96),
                       normalization_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                       corruption=None):
    dataset = wilds.get_dataset(dataset=dataset_name, root_dir=dir)
    
    # For adding coruption to the datasets
    extra_transforms = [] if corruption is None else [
        transforms.Resize(size=(224,224)),
        converters.PilToNumpy(),
        transforms.RandomChoice(corruption_transforms(corruption)),
        converters.NumpyToPil()
    ]
    
    transform = transforms.Compose([
        *extra_transforms,
        transforms.Resize(size=resize),
        transforms.ToTensor(),
        normalization_transform if  dataset_name in ["camelyon17"] else transforms.Lambda(lambda x: standardize(x))
    ])
    return dataset.get_subset(split, transform=transform)