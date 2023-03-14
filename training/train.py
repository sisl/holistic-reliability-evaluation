import sys
import os
import yaml
import argparse

# Load pytorch lightning stuff for the trainer setup
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything
import wandb

# Load the function to get the model type from the algorithm string
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from hre_model import get_model_class

# Function to load in the configuration file(s)
def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--phase", default="train")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# Load all of the configs, later ones taking precendence over earlier ones. 
config = load_config(args.config)
config["phase"] = args.phase
config["seed"] = args.seed

# Construct the save directories. 
savedir = os.path.join(config["save_folder"], config["train_dataset"], config["algorithm"], config["phase"])
if not os.path.exists(savedir):
    os.makedirs(savedir)

# Build the logger
logger = WandbLogger(
        project=config["train_dataset"],
        save_dir=savedir,
        )

# print and merge the configs
config.update(wandb.config)
print("wandb config: ", wandb.config)

# Build the trainer
trainer = pl.Trainer(
    max_epochs=config["max_epochs"],
    accelerator=config["accelerator"],
    devices=config["devices"],
    logger=logger,
    enable_progress_bar=True,
    log_every_n_steps=50,
    callbacks=[
        ModelCheckpoint(
            monitor="val_performance",
            mode="max",
            filename="best_{val_performance:.2f}-{epoch}-{step}",
        ),
        ModelCheckpoint(
            monitor="val_ds_performance",
            mode="max",
            filename="best_{val_ds_performance:.2f}-{epoch}-{step}",
        ),
        ModelCheckpoint(
            monitor="val_robustness",
            mode="max",
            filename="best_{val_robustness:.2f}-{epoch}-{step}",
        ),
        ModelCheckpoint(
            monitor="val_security",
            mode="max",
            filename="best_{val_security:.2f}-{epoch}-{step}",
        ),
        ModelCheckpoint(
            monitor="val_calibration",
            mode="max",
            filename="best_{val_calibration:.2f}-{epoch}-{step}",
        ),
        ModelCheckpoint(
            monitor="val_ood_detection",
            mode="max",
            filename="best_{val_ood_detection:.2f}-{epoch}-{step}",
        ),
        ModelCheckpoint(
            monitor="hre_score",
            mode="max",
            filename="best_{hre_score:.2f}-{epoch}-{step}",
        ),
        ModelCheckpoint(save_top_k=-1),  # This saves a model after each epoch
    ],
)

# Set the global seed for reporducibility
seed_everything(config["seed"], workers=True)

# Build the pytorch-lightning model based on the type of training we want to do
model = get_model_class(config["algorithm"])(config)

# Train the model 
trainer.fit(model)