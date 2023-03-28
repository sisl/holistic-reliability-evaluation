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

def init_fine_tune(model, n_classes):
    # Start by freezing all the layers
    for param in model.parameters():
        param.requires_grad = False
        
    # replace the classifier
    if isinstance(model, torchvision.models.DenseNet):
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
        for param in model.classifier.parameters():
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
        self.train_transforms = [get_predefined_transforms(config["train_transforms"], config)]
        self.eval_transforms = [get_predefined_transforms(config["eval_transforms"], config)]
        both_transforms = [tfs.Compose([*self.train_transforms, *self.eval_transforms])]
        
        # Load the train dataset
        self.train_dataset = load_dataset(data_dir, config["train_dataset"], self.size, self.n_channels, both_transforms)

        # Build the val and test datasets
        val = load_dataset(data_dir, config["val_id_dataset"], self.size, self.n_channels, self.eval_transforms)
        val_ds_datasets = [
            load_dataset(data_dir, name, self.size, self.n_channels, self.eval_transforms) for name in config["val_ds_datasets"]
        ]
        val_ood_datasets = [
            load_dataset(data_dir, name, self.size, self.n_channels, self.eval_transforms) for name in config["val_ood_datasets"]
        ]
        self.val_datasets = HREDatasets(
            val, val_ds_datasets, val_ood_datasets, length=config["val_dataset_length"]
        )

        test = load_dataset(data_dir, config["test_id_dataset"], self.size, self.n_channels, self.eval_transforms)
        test_ds_datasets = [
            load_dataset(data_dir, name, self.size, self.n_channels, self.eval_transforms) for name in config["test_ds_datasets"]
        ]
        test_ood_datasets = [
            load_dataset(data_dir, name, self.size, self.n_channels, self.eval_transforms) for name in config["test_ood_datasets"]
        ]
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
        self.val_batch_size = config["val_batch_size"]
        self.train_batch_size = config["train_batch_size"]
        self.optimizer = optimizer_options[config["optimizer"]]
        self.lr = config["lr"]

    ## Pytorch Lightning functions
    def configure_optimizers(self):
        # TODO: Learning rate scheduler?

        return self.optimizer(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

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
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_datasets,
            batch_size=self.val_batch_size,
        )

    ## The following functions define the various reliability metrics and should either be implmented by a sublcass, or use members that should be supplied by a subclass
    def performance(self, batch):
        x, y = batch[0], batch[1]
        pred = self(x)
        return self.performance_metric(pred, y).item()

    def adversarial_performance(self, batch):
        raise NotImplementedError("Must be implemented by subclass")

    def calibration(self, batch):
        raise NotImplementedError("Must be implemented by subclass")

    # We use AUROC as the default OOD detection metric
    def ood_detection(self, id_batch, ood_batches):
        metrics = ood.utils.OODMetrics()
        target = torch.ones(id_batch[0].shape[0])
        metrics.update(self.ood_detector(id_batch[0]), target)
        for ood_batch in ood_batches:
            metrics.update(self.ood_detector(ood_batch[0]), -1 * target)
        ood_metrics = metrics.compute()
        auroc = ood_metrics["AUROC"]
        return max(auroc, 1-auroc) # We could threshold in either direction, so we take the best outcome

    ## Thesese functions use the above functions to compute the HRE score
    def performance_info(self, id_batch, prefix):
        perf = self.performance(id_batch)
        perf_norm = (perf - self.min_performance) / (self.max_performance - self.min_performance)
        return {prefix + "_performance": perf, prefix + "_performance_norm": perf_norm}

    def robustness_info(self, ds_batches, id_performance, prefix, ds_names):
        if id_performance == 0:
            id_performance = 1e-6
        results = {prefix + "_robustness": 0.0, prefix + "_ds_performance": 0.0}
        for name, batch in zip(ds_names, ds_batches):
            performance = self.performance(batch)
            robustness = min(1.0, performance / id_performance)
            results[name + "_performance"] = performance
            results[name + "_robustness"] = robustness
            results[prefix + "_ds_performance"] += performance / len(ds_batches)
            results[prefix + "_robustness"] += robustness / len(ds_batches)

        return results

    def security_info(self, id_batch, id_performance, prefix, id_name):
        if id_performance == 0:
            id_performance = 1e-6
            
        adv_performance = self.adversarial_performance(id_batch)
        security = min(1.0, adv_performance / id_performance)
        return {id_name + "_adv_performance": adv_performance,
                prefix + "_security": security}

    def calibration_info(self, id_batch, ds_batches, prefix, id_name, ds_names):
        results = {prefix + "_calibration": 0.0}
        for name, batch in zip([id_name, *ds_names], [id_batch, *ds_batches]):
            calibration = self.calibration(batch)
            results[name + "_calibration"] = calibration
            results[prefix + "_calibration"] += calibration / (len(ds_batches) + 1)

        return results

    def ood_detection_info(self, id_batch, ood_batches, prefix, ood_names):
        results = {prefix + "_ood_detection": self.ood_detection(id_batch, ood_batches)}
        # for name, batch in zip(ood_names, ood_batches):
        #     ood_detection = self.ood_detection(id_batch, batch)
        #     results[name + "_ood_detection"] = ood_detection
        #     results[prefix + "_ood_detection"] += ood_detection / len(ood_batches)

        return results

    def hre_info(self, hrebatch, prefix, id_name, ds_names, ood_names):
        performance_results = self.performance_info(hrebatch["id"], prefix)
        id_perf = performance_results[prefix + "_performance"]
        robustness_results = self.robustness_info(
            hrebatch["ds"], id_perf, prefix, ds_names
        )
        security_results = self.security_info(
            hrebatch["id"], id_perf, prefix, id_name
        )
        calibration_results = self.calibration_info(
            hrebatch["id"], hrebatch["ds"], prefix, id_name, ds_names
        )
        ood_results = self.ood_detection_info(
            hrebatch["id"], hrebatch["ood"], prefix, ood_names
        )
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
            if "pretrained_weights" not in config or config["pretrained_weights"] == "none":
                self.model = get_model(config["model"], num_classes=self.n_classes)
            elif config["pretrained_weights"] == "default":
                weights = get_model_weights(config["model"]).DEFAULT
                self.model = get_model(config["model"], weights=weights)
                init_fine_tune(self.model, self.n_classes)
            else:
                raise ValueError("Unknown pretrained weights option {}".format(config["pretrained_weights"]))
        else:
            self.model=model

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
    def adversarial_performance(self, batch):
        if self.num_adv == 0:
            return -1.0
        
        adversary = AutoAttack(self.model, device=self.device)
        
        # Set the target classes to not exceed the number of remaining classes
        adversary.fab.n_target_classes = min(9, self.n_classes-1)
        adversary.apgd_targeted.n_target_classes = min(9, self.n_classes-1)
        
        # The targets pgd attack uses a loss that is not compatible with less than 4 classes
        if self.n_classes < 4:
            adversary.attacks_to_run = ['apgd-ce', 'fab-t', 'square']
        
        # Only use self.num_adv samples
        x = batch[0][:self.num_adv, ...]
        y = batch[1][:self.num_adv]
        
        # Run the attack and comput adversarial accuracy
        _, yadv = adversary.run_standard_evaluation(x, y, return_labels=True)
        adv_acc = sum(y == yadv).cpu().item() / y.size(0)
        return adv_acc


    # For classification we use ECE as the default calibration metric (and need to softmax it)
    def calibration(self, batch):
        x, y = batch[0], batch[1]
        logits = self(x)
        ece = self.calibration_metric(logits.softmax(1), y).item()
        return 1 - ece / 0.5
