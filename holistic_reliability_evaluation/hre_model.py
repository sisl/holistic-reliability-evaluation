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
import open_clip
import torchattacks

import multiprocessing
import sys, os

sys.path.append(os.path.dirname(__file__))
from hre_datasets import HREDatasets, load_dataset
import mae
from mae.models_vit import VisionTransformer as MAE_VisionTransformer
from utils import *

# Set the precision to speed things up a bit
torch.set_float32_matmul_precision("medium")

torch.multiprocessing.set_sharing_strategy("file_system")

dataset_cache = {} # used for caching datsets over multiple runs

def swap_classifier(model, n_classes):
    if isinstance(model, torchvision.models.DenseNet):
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    if (
        isinstance(model, torchvision.models.EfficientNet)
        or isinstance(model, torchvision.models.ConvNeXt)
        or isinstance(model, torchvision.models.MaxVit)
    ):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    if isinstance(model, torchvision.models.SwinTransformer):
        model.head = nn.Linear(model.head.in_features, n_classes)
    if isinstance(model, torchvision.models.VisionTransformer):
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, n_classes)
    if isinstance(model, torchvision.models.ResNet):
        model.fc = nn.Linear(model.fc.in_features, n_classes)


def construct_model(config):
    n_classes = config["n_classes"]
    model = config["model"]
    weights = config["pretrained_weights"]
    if "model_source" not in config or config["model_source"] == "torchvision":
        if weights is None:
            model = get_model(model, num_classes=n_classes)
        else:
            weights = getattr(get_model_weights(model), weights)
            model = get_model(model, weights=weights)
            swap_classifier(model, n_classes)
    elif config["model_source"] == "open_clip":
        clip = open_clip.create_model(model_str_to_clip(model), weights)
        classifier = nn.Linear(clip.token_embedding.embedding_dim, n_classes)
        model = CLIPClassifier(clip, classifier)
    elif config["model_source"] == "mae":
        assert weights == "DEFAULT"
        model = load_mae(model_str_to_mae(model), n_classes)
    else:
        raise ValueError(f"Unknown model source {config['model_source']}")
    return model


def freeze_weights(model, nlayers):
    assert nlayers >= 1

    # Start by freezing all the layers
    for param in model.parameters():
        param.requires_grad = False

    # Then for each model type, we unfreeze the appropriate layers
    if isinstance(model, torchvision.models.DenseNet):
        raise NotImplementedError("ResNet not implemented yet")
        # for param in model.classifier.parameters():
        #     param.requires_grad = True
    if isinstance(model, torchvision.models.VisionTransformer):
        for param in model.heads[-1].parameters():
            param.requires_grad = True
        for i in range(min(nlayers, len(model.encoder.layers)) - 1):
            for param in model.encoder.layers[-(i + 1)].parameters():
                param.requires_grad = True
    if isinstance(model, torchvision.models.ResNet):
        raise NotImplementedError("ResNet not implemented yet")
        # for param in model.fc.parameters():
        #     param.requires_grad = True
    if isinstance(model, CLIPClassifier):
        for param in model.classifier.parameters():
            param.requires_grad = True
        for i in range(min(nlayers, len(model.clip.visual.transformer.resblocks)) - 1):
            for param in model.clip.visual.transformer.resblocks[-(i + 1)].parameters():
                param.requires_grad = True
    if isinstance(model, MAE_VisionTransformer):
        for param in model.head.parameters():
            param.requires_grad = True
        for i in range(min(nlayers, len(model.blocks)) - 1):
            for param in model.blocks[-(i + 1)].parameters():
                param.requires_grad = True
    if isinstance(model, torchvision.models.MaxVit):
        for param in model.classifier.parameters():
            param.requires_grad = True
        for i in range(min(nlayers, len(model.blocks)) - 1):
            index = min(i+1, len(model.blocks))
            for param in model.blocks[-(i + 1)].parameters():
                param.requires_grad = True
    if isinstance(model, torchvision.models.SwinTransformer):
        for param in model.head.parameters():
            param.requires_grad = True
        for i in range(min(nlayers, len(model.features)) - 1):
            for param in model.features[-(i + 1)].parameters():
                param.requires_grad = True
    if isinstance(model, torchvision.models.ConvNeXt) or isinstance(
        model, torchvision.models.EfficientNet
    ):
        for param in model.classifier.parameters():
            param.requires_grad = True
        for i in range(min(nlayers, len(model.features)) - 1):
            for param in model.features[-(i + 1)].parameters():
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
        # self.save_hyperparameters()

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
        dataset_name = ''.join([config["train_dataset"], *config["train_transforms"], *config["eval_transforms"]])
        if dataset_name in dataset_cache:
            self.train_dataset = dataset_cache[dataset_name]
            print("Using cached train dataset")
        else:
            print("Loading train dataset...")
            self.train_dataset = load_dataset(
                data_dir,
                config["train_dataset"],
                self.size,
                self.n_channels,
                both_transforms,
            )
            dataset_cache[dataset_name] = self.train_dataset

        # Build the val and test datasets
        args = {
            "size": self.size,
            "n_channels": self.n_channels,
            "transforms": self.eval_transforms,
        }

        # Validation datasets
        val_length = config["val_dataset_length"]
        val_args = {**args, "length": val_length}
        
        ## Load the id-val dataset
        val_id_dataset_name = ''.join([str(val_length), *config["eval_transforms"], config["val_id_dataset"]])
        if val_id_dataset_name in dataset_cache:
            print("Using cached val id dataset")
            self.val_id = dataset_cache[val_id_dataset_name]
        else:
            print("Loading id-val dataset...")
            self.val_id = load_dataset(
                data_dir,
                config["val_id_dataset"],
                **val_args,
            )
            dataset_cache[val_id_dataset_name] = self.val_id
        
        val_dataset_names = ''.join([str(val_length), *config["eval_transforms"], config["val_id_dataset"], *config["val_ds_datasets"], *config["val_ood_datasets"]])
        if val_dataset_names in dataset_cache:
            self.val_datasets = dataset_cache[val_dataset_names]
            print("Using cached val datasets")
        else:
            val_ds_datasets = []
            for name in config["val_ds_datasets"]:
                print("Loading val ds dataset: ", name, "...")
                val_ds_datasets.append(load_dataset(data_dir, name, **val_args))
            val_ood_datasets = []
            for name in config["val_ood_datasets"]:
                print("Loading val ood dataset: ", name, "...")
                val_ood_datasets.append(load_dataset(data_dir, name, **val_args))

            self.val_datasets = HREDatasets(
                self.val_id, val_ds_datasets, val_ood_datasets, length=val_length
            )
            dataset_cache[val_dataset_names] = self.val_datasets

        # Test datasets
        test_length = config["test_dataset_length"]
        test_args = {**args, "length": test_length}
        
        test_dataset_names =  ''.join([str(test_length), *config["eval_transforms"], config["test_id_dataset"], *config["test_ds_datasets"], *config["test_ood_datasets"]])
        if test_dataset_names in dataset_cache:
            self.test_datasets = dataset_cache[test_dataset_names]
            print("Using cached test datasets")
        else:
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
            dataset_cache[test_dataset_names] = self.test_datasets

        # Set the HRE parameters and weights
        self.min_performance = config["min_performance"]
        self.max_performance = config["max_performance"]
        self.num_adv = config["num_adv"]
        
        # HRE Weights
        self.w_perf = config["w_perf"]
        self.w_rob = config["w_rob"]
        self.w_sec = config["w_sec"]
        self.w_cal = config["w_cal"]
        self.w_oodd = config["w_oodd"]
        self.normalize_hre_weights()

        # Set up remaining configuration parameters
        self.num_workers = min(multiprocessing.cpu_count(), config["max_num_workers"])
        self.batch_size = config["batch_size"]
        self.optimizer = optimizer_options[config["optimizer"]]
        self.lr = config["lr"]
        
        # Setup temperature for temp scaling
        self.T = 1
        if "calibration_method" not in config or config["calibration_method"] == "none":
            self.temperature_scale=False
        elif config["calibration_method"] == "temperature_scaling":
            print("Using temperature scaling")
            self.temperature_scale=True

    def normalize_hre_weights(self):
        total_weight = (
            self.w_perf
            + self.w_rob
            + self.w_sec
            + self.w_cal
            + self.w_oodd
        )
        self.w_perf /= total_weight
        self.w_rob /= total_weight
        self.w_sec /= total_weight
        self.w_cal /= total_weight
        self.w_oodd /= total_weight
        
    ## Pytorch Lightning functions
    def configure_optimizers(self):
        return self.optimizer(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )


    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        if self.adversarial_training_method != None:
            dev = x.device
            x = self.adversarial_training_method(x, y).to(dev)
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
        pred = self.forward(x)
        return {"pred": pred, "y": y}

    def adversarial_predictions(self, batch):
        raise NotImplementedError("Must be implemented by subclass")

    def calibration(self, batch):
        raise NotImplementedError("Must be implemented by subclass")

    # We use AUROC as the default OOD detection metric
    def ood_detection(self):
        res = {}
        for metric, ood_key in zip(self.metrics, self.ood_detectors):
            ood_metrics = metric.compute()
            auroc = ood_metrics["AUROC"]
            # We could threshold in either direction, so we take the best outcome
            res[ood_key] = max(auroc, 1 - auroc)
        
        return res

    def compute_all_needed_outputs(self, batch, batch_idx):
        if batch_idx==0:
            # Deal with calibration
            if self.temperature_scale:
                self.update_T()
            else:
                self.T = 1
                
            # Fit any OOD detectors
            # This collate function is able to ignore the iwilds metadata
            def custom_collate_fn(batch):
                x_batch, y_batch = zip(*[(x, y) for x, y, _ in batch])
                return torch.stack(x_batch), torch.stack(y_batch)
            
            d = DataLoader(self.val_id, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=custom_collate_fn)
            for ood_key in self.ood_detectors:
                self.ood_detectors[ood_key].fit(d, device=self.device)
            
            
        with torch.inference_mode():
            # Compute predictions on id and ds datasets
            predictions = {
                "id": self.predictions(batch["id"]),
                "ds": [self.predictions(b) for b in batch["ds"]],
            }

        # OOD detection
        if batch_idx == 0:
            self.metrics = [ood.utils.OODMetrics() for _ in range(len(self.ood_detectors))]
            
        pos_target = torch.ones(batch["id"][0].shape[0])
        for metric, ood_key in zip(self.metrics, self.ood_detectors):
            ood_detector = self.ood_detectors[ood_key]
            metric.update(ood_detector(batch["id"][0]), pos_target)
            for ood_batch in batch["ood"]:
                neg_target = -1 * torch.ones(ood_batch[0].shape[0])
                metric.update(ood_detector(ood_batch[0]), neg_target)

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
        
    # From: https://towardsdatascience.com/neural-network-calibration-using-pytorch-c44b7221a61
    def update_T(self):
        validation_data = self.val_dataloader()
        temperature = nn.Parameter(torch.ones(1).cuda())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')
        device = self.device

        logits_list = []
        labels_list = []

        for hrebatch in validation_data:
            batch = hrebatch["id"]
            x, y = batch[0].to(device), batch[1].to(device)

            self.model.eval()
            with torch.no_grad():
                logits_list.append(self.model(x))
                labels_list.append(y)

        # Create tensors
        logits_list = torch.cat(logits_list).to(device)
        labels_list = torch.cat(labels_list).to(device)

        def _eval():
            loss = criterion(logits_list/temperature, labels_list)
            loss.backward()
            return loss

        optimizer.step(_eval)
        self.T = temperature.item()
        print("self.T: ", self.T)

    def calibration_info(self, id_cal, ds_cals, prefix, id_name, ds_names):
        results = {prefix + "_calibration": 0.0}
        for name, calibration in zip([id_name, *ds_names], [id_cal, *ds_cals]):
            results[name + "_calibration"] = calibration
            results[prefix + "_calibration"] += calibration / (len(ds_cals) + 1)

        return results

    def ood_detection_info(self, ood_detection_metrics, prefix):
        meanval = 0
        results = {}
        for ood_key in ood_detection_metrics:
            val = ood_detection_metrics[ood_key]
            results[prefix + "_ood_detection_" + ood_key] = val
            meanval += val
        results[prefix + "_ood_detection"] = meanval / len(ood_detection_metrics)
        
        return results

    def hre_info(self, outputs, prefix, id_name, ds_names, ood_names):
        # ID
        id_pred = torch.cat([d["id"]["pred"] for d in outputs])
        id_y = torch.cat([d["id"]["y"] for d in outputs])
        id_perf = self.performance_metric(id_pred, id_y).item()
        performance_results = self.performance_info(id_perf, prefix)

        # DS
        ds_perfs = []
        for i in range(len(self.config[f"{prefix}_ds_datasets"])):
            pred = torch.cat([d["ds"][i]["pred"] for d in outputs])
            y = torch.cat([d["ds"][i]["y"] for d in outputs])
            perf = self.performance_metric(pred, y).item()
            ds_perfs.append(perf)
        robustness_results = self.robustness_info(ds_perfs, id_perf, prefix, ds_names)

        # Adv
        if self.num_adv > 0:
            adv_outputs = [
                out for out in filter(lambda d: "id_adv" in d.keys(), outputs)
            ]
            adv_pred = torch.cat([d["id_adv"]["pred"] for d in adv_outputs])
            adv_y = torch.cat([d["id_adv"]["y"] for d in adv_outputs])
            adv_perf = self.performance_metric(adv_pred, adv_y).item()
        else:
            adv_perf = 0.0
        security_results = self.security_info(adv_perf, id_perf, prefix, id_name)

        # Calibration
        id_cal = self.calibration(id_pred, id_y)
        ds_cals = []
        for i in range(len(self.config[f"{prefix}_ds_datasets"])):
            pred = torch.cat([d["ds"][i]["pred"] for d in outputs])
            y = torch.cat([d["ds"][i]["y"] for d in outputs])
            cal = self.calibration(pred, y)
            ds_cals.append(cal)
        calibration_results = self.calibration_info(
            id_cal, ds_cals, prefix, id_name, ds_names
        )

        # OOD detection
        ood_metrics = self.ood_detection()
        ood_results = self.ood_detection_info(ood_metrics, prefix)

        # Combined score
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
        
        # Save the hyperparams
        self.save_hyperparameters(ignore=["model"])

        # Load the model, possibly with pre-trained weights
        self.model = construct_model(config) if model is None else model

        # Freeze parameters for finetuning
        if config["freeze_weights"] and config["unfreeze_k_layers"] != "all":
            freeze_weights(self.model, config["unfreeze_k_layers"])


        if "adversarial_training_method" in config and config["adversarial_training_method"] != None:
            atk_type = getattr(torchattacks, config["adversarial_training_method"])
            try:  # if expression or string
                eps = eval(config["adversarial_training_eps"])
            except TypeError:  # if straight up number
                eps = config["adversarial_training_eps"]
            atk = atk_type(self.model, eps=eps)
            # find normalization values in `utils.py:get_predefined_transforms`
            atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            atk.set_device(torch.device('cuda'))
            self.adversarial_training_method = atk
        else:
            self.adversarial_training_method = None



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
        self.ood_detectors = {"EnergyBased" : ood.detector.EnergyBased(self.model),
                              "MaxSoftmax" : ood.detector.MaxSoftmax(self.model),
                              "MaxLogit" : ood.detector.MaxLogit(self.model),
                              "ODIN" : ood.detector.ODIN(self.model),
                            #   "Mahalanobis" : ood.detector.Mahalanobis(self.model),
                            #   "KLMatching" : ood.detector.KLMatching(self.model)
                              }
                              
                              
                              

        # By default use the cross entropy loss
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    def forward(self, x):
        return self.model(x)/self.T # self.T defaults to 1

    # By default, we use a gradient based attack
    def adversarial_predictions(self, batch):
        if self.num_adv == 0:
            return -1.0

        adversary = AutoAttack(self.model, device=self.device, eps=3/255)

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
