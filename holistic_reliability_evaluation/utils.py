import yaml
import csv

import torch
import torch.nn as nn

import torchvision.transforms as tfs
import torchvision.transforms.functional as TF
from torchvision.models import get_model_weights

# Function to load in the configuration file(s)
def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def load_results(results_dir):
    with open(results_dir + '/metrics.csv', 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
        return data


# Funtion to recursively flatten a model that has Sequential layers
def flatten_model(module):
    layers = []
    for child in module.children():
        if isinstance(child, nn.Sequential):
            layers.extend(flatten_model(child))
        else:
            layers.append(child)
    return layers

# Useful transforms
# Function to standarize a tensor (0 mean and 1 std)
def standardize_transform():
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.0] = 1.0
        return TF.normalize(x, mean, std)
    return tfs.Lambda(lambda x: standardize(x))

# Function to apply random rotations
def random_rotation_transform(angles = [0, 90, 180, 270]):
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    return tfs.Lambda(lambda x: random_rotation(x))

# Predefined sets of transforms, accesible by string
def get_predefined_transforms(transform_strings, config):
    transforms = []
    for transform_name in transform_strings:
        if transform_name == "wilds_default_normalization":
            transforms.append(tfs.Resize(tuple(config["size"])))
            transforms.append(tfs.ToTensor())
            transforms.append(tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        elif transform_name == "wilds_default_standardize":
            transforms.append(tfs.Resize(tuple(config["size"])))
            transforms.append(tfs.ToTensor())
            transforms.append(standardize_transform())
        elif transform_name == "random_rotation":
            transforms.append(random_rotation_transform())
        elif transform_name == "pretrain_default":
            # Using pre-trained transforms
            transforms.append(getattr(get_model_weights(config["model"]), config["pretrained_weights"]).transforms())
        else:
            raise ValueError(f"Unknown transform {transform_name}")
    
    print("Transforms: ", transforms)
    return tfs.Compose(transforms)
        