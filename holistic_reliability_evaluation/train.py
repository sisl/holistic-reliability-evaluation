import sys
import os
import argparse

# Load pytorch lightning stuff for the trainer setup
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything
import wandb

# Load the function to get the model type from the algorithm string
sys.path.append(os.path.join(os.path.dirname(__file__)))
from hre_model import ClassificationTask
from utils import load_config

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

csvlogger = CSVLogger(savedir, name=config["train_dataset"], version=logger.version)

# print and merge the configs
config.update(wandb.config)

# Build the trainer
trainer = pl.Trainer(
    max_epochs=config["max_epochs"],
    accelerator=config["accelerator"],
    devices=config["devices"],
    logger=[logger, csvlogger],
    enable_progress_bar=True,
    log_every_n_steps=50,
    callbacks=[
        ModelCheckpoint(
            monitor="val_performance",
            mode="max",
            filename="best_{val_performance:.2f}-{epoch}-{step}",
        ),
        # ModelCheckpoint(
        #     monitor="val_ds_performance",
        #     mode="max",
        #     filename="best_{val_ds_performance:.2f}-{epoch}-{step}",
        # ),
        # ModelCheckpoint(
        #     monitor="val_robustness",
        #     mode="max",
        #     filename="best_{val_robustness:.2f}-{epoch}-{step}",
        # ),
        # ModelCheckpoint(
        #     monitor="val_security",
        #     mode="max",
        #     filename="best_{val_security:.2f}-{epoch}-{step}",
        # ),
        # ModelCheckpoint(
        #     monitor="val_calibration",
        #     mode="max",
        #     filename="best_{val_calibration:.2f}-{epoch}-{step}",
        # ),
        # ModelCheckpoint(
        #     monitor="val_ood_detection",
        #     mode="max",
        #     filename="best_{val_ood_detection:.2f}-{epoch}-{step}",
        # ),
        # ModelCheckpoint(
        #     monitor="val_hre_score",
        #     mode="max",
        #     filename="best_{hre_score:.2f}-{epoch}-{step}",
        # ),
        ModelCheckpoint(),  # This saves a model after each epoch
    ],
)

# Set the global seed for reporducibility
seed_everything(config["seed"], workers=True)

# Build the pytorch-lightning model based on the type of training we want to do
model = ClassificationTask(config)

# Train the model 
trainer.fit(model)