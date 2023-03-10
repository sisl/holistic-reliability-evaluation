import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, CalibrationError
import pytorch_ood as ood
from torchvision.models import get_model

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from load_datasets import load_dataset
from hre_datasets import HREDatasets

# Set the precision to speed things up a bit
torch.set_float32_matmul_precision("medium")

# Options for the selection of the optimizer
optimizer_options = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
    "adamw": torch.optim.AdamW,
}

def get_model_class(name):
    if name == "erm":
        return ClassificationTask
    else:
        raise ValueError("Model name {} not recognized".format(name))


class HREModel(pl.LightningModule):
    def __init__(
        self,
        model,  # Underlying model
        config,  # Configuration file where are the model parameters are set
    ):
        super().__init__()

        # Save the hyperparameters
        self.save_hyperparameters(ignore=["model"])

        # Store model and config
        self.model = model
        self.config = config

        # Generate the datatsets
        data_dir = config["data_dir"]

        # Load the training datasets
        self.train_dataset = load_dataset(data_dir, config["train_dataset"])

        # Build the val and test datasets
        val = load_dataset(data_dir, config["val_id_dataset"])
        val_ds_datasets = [
            load_dataset(data_dir, name) for name in config["val_ds_datasets"]
        ]
        val_ood_datasets = [
            load_dataset(data_dir, name) for name in config["val_ood_datasets"]
        ]
        self.val_datasets = HREDatasets(
            val, val_ds_datasets, val_ood_datasets, length=config["val_dataset_length"]
        )

        test = load_dataset(data_dir, config["test_id_dataset"])
        test_ds_datasets = [
            load_dataset(data_dir, name) for name in config["test_ds_datasets"]
        ]
        test_ood_datasets = [
            load_dataset(data_dir, name) for name in config["test_ood_datasets"]
        ]
        self.test_datasets = HREDatasets(
            test,
            test_ds_datasets,
            test_ood_datasets,
            length=config["test_dataset_length"],
        )

        # Set the HRE weights
        total_weight = (
            config["w_perf"]
            + config["w_rob"]
            + config["w_sec"]
            + config["w_cal"]
            + config["w_oodd"]
        )
        self.w_perf = config["w_perf"] / total_weight
        self.w_rob = config["w_rob"] / total_weight
        self.w_sec = config["w_sec"] / total_weight
        self.w_cal = config["w_cal"] / total_weight
        self.w_oodd = config["w_oodd"] / total_weight

        # Set up remaining configuration parameters
        self.num_workers = config["num_workers"]
        self.val_batch_size = config["val_batch_size"]
        self.train_batch_size = config["train_batch_size"]
        self.optimizer = optimizer_options[config["optimizer"]]
        self.lr = config["lr"]

    ## Pytorch Lightning functions
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # TODO: Learning rate scheduler?
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.log("train_performance", self.performance_metric(logits, y).item())
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.log_dict(
            self.hre_info(
                val_batch,
                "val",
                self.config["val_id_dataset"],
                self.config["val_ds_datasets"],
                self.config["val_ood_datasets"],
            )
        )

    def test_step(self, test_batch, batch_idx):
        self.log_dict(
            self.hre_info(
                test_batch,
                "test",
                self.config["test_id_dataset"],
                self.config["test_ds_datasets"],
                self.config["test_ood_datasets"],
            )
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_datasets,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_datasets,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )

    ## The following functions define the various reliability metrics and should either be implmented by a sublcass, or use members that should be supplied by a subclass
    def performance(self, batch):
        x, y = batch[0], batch[1]
        pred = self(x)
        return self.performance_metric(pred, y).item()

    def security(self, batch):
        raise NotImplementedError("Must be implemented by subclass")

    def calibration(self, batch):
        raise NotImplementedError("Must be implemented by subclass")

    # We use AUROC as the default OOD detection metric
    def ood_detection(self, id_batch, ood_batch):
        metrics = ood.utils.OODMetrics()
        target = torch.ones(id_batch[0].shape[0])
        metrics.update(self.ood_detector(id_batch[0]), target)
        metrics.update(self.ood_detector(ood_batch[0]), -1 * target)
        ood_metrics = metrics.compute()
        return ood_metrics["AUROC"]

    ## Thesese functions use the above functions to compute the HRE score
    def performance_info(self, id_batch, prefix):
        return {prefix + "_performance": self.performance(id_batch)}

    def robustness_info(self, ds_batches, id_performance, prefix, ds_names):
        results = {prefix + "_robustness": 0.0, prefix + "_ds_performance": 0.0}
        for name, batch in zip(ds_names, ds_batches):
            performance = self.performance(batch)
            robustness = performance / id_performance
            results[name + "_performance"] = performance
            results[name + "_robustness"] = robustness
            results[prefix + "_ds_performance"] += performance / len(ds_batches)
            results[prefix + "_robustness"] += robustness / len(ds_batches)

        return results

    def security_info(self, id_batch, ds_batches, prefix, id_name, ds_names):
        results = {prefix + "_security": 0.0}
        for name, batch in zip([id_name, *ds_names], [id_batch, *ds_batches]):
            security = self.security(batch)
            results[name + "_security"] = security
            results[prefix + "_security"] += security / (len(ds_batches) + 1)

        return results

    def calibration_info(self, id_batch, ds_batches, prefix, id_name, ds_names):
        results = {prefix + "_calibration": 0.0}
        for name, batch in zip([id_name, *ds_names], [id_batch, *ds_batches]):
            calibration = self.calibration(batch)
            results[name + "_calibration"] = calibration
            results[prefix + "_calibration"] += calibration / (len(ds_batches) + 1)

        return results

    def ood_detection_info(self, id_batch, ood_batches, prefix, ood_names):
        results = {prefix + "_ood_detection": 0.0}
        for name, batch in zip(ood_names, ood_batches):
            ood_detection = self.ood_detection(id_batch, batch)
            results[name + "_ood_detection"] = ood_detection
            results[prefix + "_ood_detection"] += ood_detection / len(ood_batches)

        return results

    def hre_info(self, hrebatch, prefix, id_name, ds_names, ood_names):
        performance_results = self.performance_info(hrebatch["id"], prefix)
        id_perf = performance_results[prefix + "_performance"]
        robustness_results = self.robustness_info(
            hrebatch["ds"], id_perf, prefix, ds_names
        )
        security_results = self.security_info(
            hrebatch["id"], hrebatch["ds"], prefix, id_name, ds_names
        )
        calibration_results = self.calibration_info(
            hrebatch["id"], hrebatch["ds"], prefix, id_name, ds_names
        )
        ood_results = self.ood_detection_info(
            hrebatch["id"], hrebatch["ood"], prefix, ood_names
        )
        hre_score = (
            id_perf * self.w_perf
            + robustness_results[prefix + "_robustness"] * self.w_rob
            + security_results[prefix + "_security"] * self.w_sec
            + calibration_results[prefix + "_calibration"] * self.w_cal
            + ood_results[prefix + "_ood_detection"] * self.w_oodd
        )

        return {
            **performance_results,
            **robustness_results,
            **security_results,
            **calibration_results,
            **ood_results,
            "hre_score": hre_score,
        }


# Instantiate the defaults for a classification task
class ClassificationTask(HREModel):
    def __init__(self, config, *args, **kwargs):
        # TODO: Torchvision provides easy pre-trained weights

        self.n_classes = config["n_classes"]
        model = get_model(config["model"], num_classes=self.n_classes)

        # Call the HRE constuctor
        super().__init__(model, config, *args, **kwargs)

        # Set the defaults for a classification task
        # By default we set the performance metric to accuracy
        self.performance_metric = Accuracy(
            task="multiclass", num_classes=self.n_classes, top_k=1
        )

        # By default use the ECE
        self.calibration_metric = CalibrationError(
            task="multiclass", num_classes=self.n_classes
        )

        # By default use an energy based OOD detector
        self.ood_detector = ood.detector.EnergyBased(self.model)
        # By default use the cross entropy loss
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    # By default, we use a gradient based attack
    def security(self, batch):
        return 1.0  # TODO: Fill this in

    # For classification we use ECE as the default calibration metric (and need to softmax it)
    def calibration(self, batch):
        x, y = batch[0], batch[1]
        logits = self(x)
        ece = self.calibration_metric(logits.softmax(1), y).item()
        return 1 - ece / 0.5
