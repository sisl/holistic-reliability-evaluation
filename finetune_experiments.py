from holistic_reliability_evaluation.train import *
from holistic_reliability_evaluation.utils import *
import numpy as np
from collections import namedtuple

Ntrials = 20
dataset = "rxrx1"
config = load_config(f"configs/{dataset}-finetune.yml")
config["finetune_experimets"] = True

label_smoothing_sampler = lambda: np.random.choice([0.1, 0.01, 0.0]).item()
optimizer_sampler = lambda: np.random.choice(["adam", "sgd", "adamw"]).item()
batch_size_sampler = lambda: np.random.choice([12, 24, 48]).item()
lr_sampler = lambda: np.power(10.0, np.random.uniform(-4.0, -2.0)).item()
def unfreeze_k_layers_sampler(): 
    samp = np.random.choice([1, 2, 4, 8, "all"]).item()
    if samp.isnumeric():
        return int(samp)
    else:
        return samp

Setup = namedtuple("Setup", ["model_source", "model", "pretrained_weights"])

setups = [
    # Different architecures trained discriminatively
    Setup("torchvision", "efficientnet_v2_l", "IMAGENET1K_V1"),
    Setup("torchvision", "convnext_large", "IMAGENET1K_V1"),
    Setup("torchvision", "maxvit_t", "IMAGENET1K_V1"),
    Setup("torchvision", "swin_v2_b", "IMAGENET1K_V1"),
    Setup("torchvision", "vit_b_16", "IMAGENET1K_V1"),
    Setup("torchvision", "vit_l_16", "IMAGENET1K_V1"),
    # Same architecture pretrained in different ways
    Setup("torchvision", "vit_b_16", "IMAGENET1K_SWAG_LINEAR_V1"),
    Setup("torchvision", "vit_l_16", "IMAGENET1K_SWAG_LINEAR_V1"),
    Setup("torchvision", "vit_h_14", "IMAGENET1K_SWAG_LINEAR_V1"),
    Setup("open_clip", "vit_b_16", "openai"),
    Setup("open_clip", "vit_l_14", "openai"),
    Setup("open_clip", "vit_h_14", "laion2b_s32b_b79k"),
    Setup("mae", "vit_b_16", "DEFAULT"),
    Setup("mae", "vit_l_16", "DEFAULT"),
    Setup("mae", "vit_h_14", "DEFAULT"),
]
file_name = f"{dataset}_record.txt"

# Loop over hyperparameters
for trial in range(Ntrials):
    np.random.seed(trial)
    config["seed"] = trial
    config["label_smoothing"] = label_smoothing_sampler()
    config["optimizer"] = optimizer_sampler()
    config["batch_size"] = batch_size_sampler()
    config["lr"] = lr_sampler()
    config["unfreeze_k_layers"] = unfreeze_k_layers_sampler()
    
    for setup in setups:
        
        # create string to check if the setup has already been run
        str = f"{setup.model_source}_{setup.model}_{setup.pretrained_weights}_{config['optimizer']}_{config['label_smoothing']}_{config['batch_size']}_{config['seed']}_{config['lr']}_{config['unfreeze_k_layers']}"
        
        # Check if file exists
        with open(file_name, "r") as file:
            file_contents = file.read()
            if str in file_contents:
                print(f"{str} is already in the file. Skipping...")
                continue
        
        print("Trial:", trial)
        print("Running new setup: ", setup)
        # Set model_source
        config["algorithm"] = "_".join(
            [setup.model_source, setup.model, setup.pretrained_weights]
        )
        config["model_source"] = setup.model_source
        config["model"] = setup.model
        config["pretrained_weights"] = setup.pretrained_weights

        train(config)
        
        # Write the string to the file
        with open(file_name, "a") as file:
            file.write(f"{str}\n")
            print(f"{str} has been written to the file.")
