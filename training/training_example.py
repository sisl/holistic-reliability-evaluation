import math
import torch
import pytorch_lightning as lit
# from lit.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics as tm
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
import os

from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

class LitMNIST(lit.LightningModule):
    def __init__(self, config):
        super().__init__()

        # Set our init args as class attributes
        self.hidden_size = config["hidden_size"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_classes),
        )

        self.val_accuracy = tm.Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = tm.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return nn.functional.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def train_mnist_tune(config, train_data, val_data, test_data, num_epochs=10, num_gpus=0):
    model = LitMNIST(config)
    trainer = lit.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=os.getcwd(), name="", version="."),
        enable_progress_bar=False,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy"
                },
                on="validation_end")
        ])
    trainer.fit(model, train_data, val_data, test_data)
    
transform = transforms.ToTensor()
train_set = DataLoader(MNIST(root="MNIST", download=False, train=True, transform=transform))
val_set = DataLoader(MNIST(root="MNIST", download=False, train=False, transform=transform))
test_set = DataLoader(MNIST(root="MNIST", download=False, train=False, transform=transform))

config = {
    "hidden_size": tune.choice([32, 64, 128]),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64, 128]),
}


scheduler = ASHAScheduler(
    max_t=10,
    grace_period=1,
    reduction_factor=2
    )

reporter = CLIReporter(
    parameter_columns=["hidden_size", "learning_rate", "batch_size"],
    metric_columns=["loss", "mean_accuracy", "training_iteration"]
    ) 

gpus_per_trial = 0

train_fn_with_parameters = tune.with_parameters(train_mnist_tune,
                                                    num_epochs=10,
                                                    num_gpus=0,
                                                    train_data=train_set,
                                                    val_data=val_set,
                                                    test_data=test_set)
resources_per_trial = {"cpu": 1, "gpu": 0}

num_samples=10
tuner = tune.Tuner(
    tune.with_resources(
        train_fn_with_parameters,
        resources=resources_per_trial
    ),
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
        scheduler=scheduler,
        num_samples=num_samples,
    ),
    run_config=air.RunConfig(
        name="tune_mnist_asha",
        progress_reporter=reporter,
    ),
    param_space=config,
)
results = tuner.fit()

print("Best hyperparameters found were: ", results.get_best_result().config)
    
    


