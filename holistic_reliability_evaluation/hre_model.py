import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, CalibrationError
import pytorch_ood as ood
from autoattack import AutoAttack
import torchvision
from torchvision.models import get_model, get_model_weights
import torchvision.transforms as tfs

import multiprocessing
import sys, os

sys.path.append(os.path.dirname(__file__))
from hre_datasets import HREDatasets, load_dataset
from utils import flatten_model, get_predefined_transforms

# Set the precision to speed things up a bit
torch.set_float32_matmul_precision("medium")


def swap_classifier(model, n_classes):
    if isinstance(model, torchvision.models.DenseNet):
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    if isinstance(model, torchvision.models.VisionTransformer):
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, n_classes)
    if isinstance(model, torchvision.models.ResNet):
        model.fc = nn.Linear(model.fc.in_features, n_classes)


def freeze_weights(model, nlayers):
    # Start by freezing all the layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier layer
    if isinstance(model, torchvision.models.DenseNet):
        for param in model.classifier.parameters():
            param.requires_grad = True
    if isinstance(model, torchvision.models.VisionTransformer):
        assert nlayers >= 1
        # unfreeze the first layer
        for param in model.heads[-1].parameters():
            param.requires_grad = True
        # unfreeze the remaining k layers
        for i in range(nlayers - 1):
            for param in model.encoder.layers[-(i - 1)].parameters():
                param.requires_grad = True
    if isinstance(model, torchvision.models.ResNet):
        for param in model.fc.parameters():
            param.requires_grad = True


# Options for the selection of the optimizer
optimizer_options = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
    "adamw": torch.optim.AdamW,
}


class HREModel(pl.LightningModule):
    def __init__(
        self,
        config,  # Configuration file where are the model parameters are set
    ):
        super().__init__()

        # Save the hyperparameters
        self.save_hyperparameters()

        # Store config
        self.config = config

        # Generate the datatsets
        data_dir = config["data_dir"]

        # Get information on the datasets
        self.size = tuple(config["size"])
        self.n_channels = config["n_channels"]
        self.n_classes = config["n_classes"]

        # Load the transforms
        self.train_transforms = [
            get_predefined_transforms(config["train_transforms"], config)
        ]
        self.eval_transforms = [
            get_predefined_transforms(config["eval_transforms"], config)
        ]
        both_transforms = [tfs.Compose([*self.train_transforms, *self.eval_transforms])]

        # Load the train dataset
        print("Loading train dataset...")
        self.train_dataset = load_dataset(
            data_dir,
            config["train_dataset"],
            self.size,
            self.n_channels,
            both_transforms,
        )

        # Build the val and test datasets
        args = {
            "size": self.size,
            "n_channels": self.n_channels,
            "transforms": self.eval_transforms,
        }

        # Validation datasets
        val_length = config["val_dataset_length"]
        val_args = {**args, "length": val_length}
        print("Loading id-val dataset...")
        val = load_dataset(
            data_dir,
            config["val_id_dataset"],
            **val_args,
        )
        val_ds_datasets = []
        for name in config["val_ds_datasets"]:
            print("Loading val ds dataset: ", name, "...")
            val_ds_datasets.append(load_dataset(data_dir, name, **val_args))
        val_ood_datasets = []
        for name in config["val_ood_datasets"]:
            print("Loading val ood dataset: ", name, "...")
            val_ood_datasets.append(load_dataset(data_dir, name, **val_args))

        self.val_datasets = HREDatasets(
            val, val_ds_datasets, val_ood_datasets, length=val_length
        )

        # Test datasets
        test_length = config["test_dataset_length"]
        test_args = {**args, "length": test_length}

        print("Loading id-test dataset...")
        test = load_dataset(
            data_dir,
            config["test_id_dataset"],
            **test_args,
        )

        test_ds_datasets = []
        for name in config["test_ds_datasets"]:
            print("Loading test ds dataset: ", name, "...")
            test_ds_datasets.append(load_dataset(data_dir, name, **test_args))

        test_ood_datasets = []
        for name in config["test_ood_datasets"]:
            print("Loading test ood dataset: ", name, "...")
            test_ood_datasets.append(load_dataset(data_dir, name, **test_args))

        self.test_datasets = HREDatasets(
            test,
            test_ds_datasets,
            test_ood_datasets,
            length=config["test_dataset_length"],
        )

        # Set the HRE parameters and weights
        self.min_performance = config["min_performance"]
        self.max_performance = config["max_performance"]
        self.num_adv = config["num_adv"]
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
        self.num_workers = min(multiprocessing.cpu_count(), config["max_num_workers"])
        self.batch_size = config["batch_size"]
        self.optimizer = optimizer_options[config["optimizer"]]
        self.lr = config["lr"]

    ## Pytorch Lightning functions
    def configure_optimizers(self):
        # TODO: Learning rate scheduler?

        return self.optimizer(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.log("train_performance", self.performance_metric(logits, y).item())
        return loss

    def validation_step(self, val_batch, batch_idx):
        return self.compute_all_needed_outputs(val_batch, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):
        self.log_dict(
            self.hre_info(
                validation_step_outputs,
                "val",
                self.config["val_id_dataset"],
                self.config["val_ds_datasets"],
                self.config["val_ood_datasets"],
            )
        )

    def test_step(self, test_batch, batch_idx):
        return self.compute_all_needed_outputs(test_batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        self.log_dict(
            self.hre_info(
                test_step_outputs,
                "test",
                self.config["test_id_dataset"],
                self.config["test_ds_datasets"],
                self.config["test_ood_datasets"],
            )
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    ## The following functions define the various reliability metrics and should either be implmented by a sublcass, or use members that should be supplied by a subclass
    def predictions(self, batch):
        x, y = batch[0], batch[1]
        pred = self.model(x)
        return {"pred": pred, "y": y}

    def adversarial_predictions(self, batch):
        raise NotImplementedError("Must be implemented by subclass")

    def calibration(self, batch):
        raise NotImplementedError("Must be implemented by subclass")

    # We use AUROC as the default OOD detection metric
    def ood_detection(self):
        ood_metrics = self.metrics.compute()
        auroc = ood_metrics["AUROC"]
        # We could threshold in either direction, so we take the best outcome
        return max(auroc, 1 - auroc)

    def compute_all_needed_outputs(self, batch, batch_idx):
        with torch.inference_mode():
            # Compute predictions on id and ds datasets
            predictions = {
                "id": self.predictions(batch["id"]),
                "ds": [self.predictions(b) for b in batch["ds"]],
            }

            # OOD detection
            if batch_idx == 0:
                self.metrics = ood.utils.OODMetrics()

            target = torch.ones(batch["id"][0].shape[0])
            self.metrics.update(self.ood_detector(batch["id"][0]), target)
            for ood_batch in batch["ood"]:
                self.metrics.update(self.ood_detector(ood_batch[0]), -1 * target)

        # Compute adversarial predictions
        if batch_idx * self.batch_size < self.config["num_adv"]:
            y_adv = self.adversarial_predictions(batch["id"])
            predictions["id_adv"] = y_adv

        return predictions

    ## Thesese functions use the above functions to compute the HRE score
    def performance_info(self, perf, prefix):
        perf_norm = (perf - self.min_performance) / (
            self.max_performance - self.min_performance
        )
        return {prefix + "_performance": perf, prefix + "_performance_norm": perf_norm}

    def robustness_info(self, ds_perfs, id_performance, prefix, ds_names):
        if id_performance == 0:
            id_performance = 1e-6
        results = {prefix + "_robustness": 0.0, prefix + "_ds_performance": 0.0}
        for name, perf in zip(ds_names, ds_perfs):
            robustness = min(1.0, perf / id_performance)
            results[name + "_performance"] = perf
            results[name + "_robustness"] = robustness
            results[prefix + "_ds_performance"] += perf / len(ds_perfs)
            results[prefix + "_robustness"] += robustness / len(ds_perfs)

        return results

    def security_info(self, adv_performance, id_performance, prefix, id_name):
        if id_performance == 0:
            id_performance = 1e-6

        security = min(1.0, adv_performance / id_performance)
        return {
            id_name + "_adv_performance": adv_performance,
            prefix + "_security": security,
        }

    def calibration_info(self, id_cal, ds_cals, prefix, id_name, ds_names):
        results = {prefix + "_calibration": 0.0}
        for name, calibration in zip([id_name, *ds_names], [id_cal, *ds_cals]):
            results[name + "_calibration"] = calibration
            results[prefix + "_calibration"] += calibration / (len(ds_cals) + 1)

        return results

    def ood_detection_info(self, ood_detection_metric, prefix):
        return {prefix + "_ood_detection": ood_detection_metric}

    def hre_info(self, outputs, prefix, id_name, ds_names, ood_names):
        # ID
        pred = torch.cat([d["id"]["pred"] for d in outputs])
        y = torch.cat([d["id"]["y"] for d in outputs])
        id_perf = self.performance_metric(pred, y).item()
        performance_results = self.performance_info(id_perf, prefix)

        # DS
        ds_perfs = []
        for i in range(len(self.config[f"val_ds_datasets"])):
            pred = torch.cat([d["ds"][i]["pred"] for d in outputs])
            y = torch.cat([d["ds"][i]["y"] for d in outputs])
            perf = self.performance_metric(pred, y).item()
            ds_perfs.append(perf)
        robustness_results = self.robustness_info(ds_perfs, id_perf, prefix, ds_names)

        # Adv
        adv_outputs = [out for out in filter(lambda d: "id_adv" in d.keys(), outputs)]
        adv_pred = torch.cat([d["id_adv"]["pred"] for d in adv_outputs])
        adv_y = torch.cat([d["id_adv"]['y'] for d in adv_outputs])
        adv_perf = self.performance_metric(adv_pred, adv_y).item()
        security_results = self.security_info(adv_perf, id_perf, prefix, id_name)

        # Calibration
        id_cal = self.calibration(pred, y)
        ds_cals = []
        for i in range(len(self.config[f"val_ds_datasets"])):
            pred = torch.cat([d["ds"][i]["pred"] for d in outputs])
            y = torch.cat([d["ds"][i]["y"] for d in outputs])
            cal = self.calibration(pred, y)
            ds_cals.append(cal)
        calibration_results = self.calibration_info(
            id_cal, ds_cals, prefix, id_name, ds_names
        )

        # OOD detection
        ood_metric = self.ood_detection()
        ood_results = self.ood_detection_info(ood_metric, prefix)

        hre_score = (
            performance_results[prefix + "_performance_norm"] * self.w_perf
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
            prefix + "_hre_score": hre_score,
        }


# Instantiate the defaults for a classification task
class ClassificationTask(HREModel):
    def __init__(self, config, model=None, *args, **kwargs):
        # Call the HRE constuctor
        super().__init__(config, *args, **kwargs)

        # Load the model, possibly with pre-trained weights
        if model is None:
            if (
                "pretrained_weights" not in config
                or config["pretrained_weights"] == "none"
            ):
                self.model = get_model(config["model"], num_classes=self.n_classes)
            else:
                weights = getattr(
                    get_model_weights(config["model"]), config["pretrained_weights"]
                )
                self.model = get_model(config["model"], weights=weights)
                swap_classifier(self.model, self.n_classes)
                if config["freeze_weights"]:
                    freeze_weights(self.model, config["unfreeze_k_layers"])
        else:
            self.model = model

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

    def forward(self, x):
        return self.model(x)

    # By default, we use a gradient based attack
    def adversarial_predictions(self, batch):
        if self.num_adv == 0:
            return -1.0

        adversary = AutoAttack(self.model, device=self.device)

        # Set the target classes to not exceed the number of remaining classes
        adversary.fab.n_target_classes = min(9, self.n_classes - 1)
        adversary.apgd_targeted.n_target_classes = min(9, self.n_classes - 1)

        # The targets pgd attack uses a loss that is not compatible with less than 4 classes
        if self.n_classes < 4:
            adversary.attacks_to_run = ["apgd-ce", "fab-t", "square"]

        # Run the attack and return the perturbed labels
        x, y = batch[0], batch[1]
        _, yadv = adversary.run_standard_evaluation(x, y, return_labels=True)
        return {"pred": yadv, "y": y}

    # For classification we use ECE as the default calibration metric (and need to softmax it)
    def calibration(self, pred, y):
        ece = self.calibration_metric(pred.softmax(1), y).item()
        return 1 - ece / 0.5
