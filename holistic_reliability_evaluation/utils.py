import yaml
import csv
import requests
from collections import namedtuple

import torch
import torch.nn as nn

import torchvision.transforms as tfs
import torchvision.transforms.functional as TF
from torchvision.models import get_model_weights
from open_clip import create_model_and_transforms

import sys, os
sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.join(os.path.dirname(__file__), "mae/util/"))
# sys.path.append(os.path.join(os.path.dirname(__file__), "mae/"))
import mae
import mae.models_vit
from mae.util.datasets import build_transform
from mae.util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

class CLIPClassifier(nn.Module):
    def __init__(self, clip, classifier):
        super(CLIPClassifier, self).__init__()
        self.clip = clip
        self.classifier = classifier

    def forward(self, x):
        z = self.clip.encode_image(x)
        return self.classifier(z)

def mae_url(model):
    if model == "vit_base_patch16":
        checkpoint_url = "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth"
    elif model == "vit_large_patch16":
        checkpoint_url = "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth"
    elif model == "vit_huge_patch14":
        checkpoint_url = "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth"
    else:
        raise ValueError(f"Model {model} not supported.")
    return checkpoint_url

def load_mae(model_name, n_classes, global_pool=False, drop_path=0.1, temp_dir="."):
    checkpoint_url = mae_url(model_name)
    checkpoint_path = os.path.join(temp_dir, f"{model_name}.pth")

    download_large_file(checkpoint_url, checkpoint_path)

    model = mae.models_vit.__dict__[model_name](
            num_classes=n_classes,
            drop_path_rate=drop_path,
            global_pool=global_pool,
        )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    model.load_state_dict(checkpoint_model, strict=False)

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)
    return model

def model_str_to_mae(s):
    # Split the string into three parts based on the underscores
    parts = s.split('_')
    # Check the middle letter and replace it with the corresponding word
    if parts[1] == 'b':
        parts[1] = 'base'
    elif parts[1] == 'l':
        parts[1] = 'large'
    elif parts[1] == 'h':
        parts[1] = 'huge'
    # Join the parts back together, insert "patches", and return the result
    return f"{parts[0]}_{parts[1]}_patch{parts[2]}"

def model_str_to_clip(s):
    # Split the string into three parts based on the underscores
    parts = s.split('_')
    # Convert the middle letter to uppercase and add hyphens before and after it
    middle_letter = parts[1].upper()
    # Join the parts back together with hyphens and return the result
    return f"ViT-{middle_letter}-{parts[2]}"
    
def download_large_file(url, output_file, chunk_size=8192):
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping download.")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # Filter out keep-alive new chunks
                f.write(chunk)
    print(f"{output_file} downloaded successfully.")

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
            model = config["model"]
            weights = config["pretrained_weights"]
            if config["model_source"] == "torchvision":
                tf = getattr(get_model_weights(model), weights).transforms()
            elif config["model_source"] == "open_clip":
                tf = create_model_and_transforms(model_str_to_clip(model), weights)[2]
            elif config["model_source"] == "mae":
                assert config["size"][0] == config["size"][1]
                Args = namedtuple("Args", ["input_size"])
                args = Args(config["size"][0])
                tf = build_transform(False, args)
            else:
                raise ValueError(f"Unknown model source {config['model_source']}")
            transforms.append(tf)
        else:
            raise ValueError(f"Unknown transform {transform_name}")
    
    print("Transforms: ", transforms)
    return tfs.Compose(transforms)
        