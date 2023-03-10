import wilds
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import random_split, Dataset, Subset


# This comes from imagenet_cbar
# import converters
# from transform_finder import transform_dict, build_transform
    
def load_dataset(data_dir, name):
    if name == "camelyon17-train":
        return load_wilds_dataset("camelyon17", data_dir, split="train")
    elif name == "camelyon17-id_val":
        id_val_dataset = load_wilds_dataset("camelyon17", data_dir, split="id_val")
        val, test = random_split(id_val_dataset, [0.5, 0.5], generator=torch.Generator().manual_seed(0))
        return val
    elif name == "camelyon17-val":
        return load_wilds_dataset("camelyon17", data_dir, split="val")
    elif name == "camelyon17-id_test":
        id_val_dataset = load_wilds_dataset("camelyon17", data_dir, split="id_val")
        val, test = random_split(id_val_dataset, [0.5, 0.5], generator=torch.Generator().manual_seed(0))
        return test
    elif name == "camelyon17-test":
        return load_wilds_dataset("camelyon17", data_dir, split="test")
    elif name == "rxrx1-id_test":
        return load_wilds_dataset("rxrx1", data_dir, split="id_test")
    elif name == "gaussian_noise-(96x96)":
        return random_noise_dataset(items=1000, size=(96,96), channels=3)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

def random_noise_dataset(items=1000, size=(96,96), channels=3):
    Xtrain = torch.randn(items, channels, *size)
    ytrain = torch.randn(items, 1)
    md = torch.randn(items, 1)
    return torch.utils.data.TensorDataset(Xtrain, ytrain, md) 


def standardize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(1, 2))
    std = x.std(dim=(1, 2))
    std[std == 0.] = 1.
    return TF.normalize(x, mean, std)

def corruption_transforms(severity, types):
    return [build_transform(name, severity, 'imagenet')  for name in types]
    # return [build_transform(name, severity, 'imagenet')  for name in set(transform_dict.keys()).difference({'color_balance'})]


def load_wilds_dataset(dataset_name, 
                       dir,
                       split="test",
                       resize=(96,96),
                       normalization_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                       corruption=None,
                       corruption_types=['pseudocolor', 'single_frequency_greyscale', 'perspective_no_bars', 'lines', 'technicolor', 'chromatic_abberation', 'bleach_bypass', 'hue_shift', 'cocentric_sine_waves', 'transverse_chromatic_abberation', 'quadrilateral_no_bars']):
    dataset = wilds.get_dataset(dataset=dataset_name, root_dir=dir)
    
    # For adding coruption to the datasets
    extra_transforms = [] if corruption is None else [
        transforms.Resize(size=(224,224)),
        converters.PilToNumpy(),
        transforms.RandomChoice(corruption_transforms(corruption, corruption_types)),
        converters.NumpyToPil()
    ]
    
    transform = transforms.Compose([
        *extra_transforms,
        transforms.Resize(size=resize),
        transforms.ToTensor(),
        normalization_transform if  dataset_name in ["camelyon17"] else transforms.Lambda(lambda x: standardize(x))
    ])
    return dataset.get_subset(split, transform=transform)