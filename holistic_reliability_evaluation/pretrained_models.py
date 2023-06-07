import torchvision
import torch
import torch.nn as nn

import yaml

import sys, os

thisfilepath = os.path.dirname(__file__)
sys.path.append(thisfilepath)
from utils import load_config
from hre_model import ClassificationTask


# This is used to load models that *we* have pretrained, not those from wilds.
def load_model_descriptions(results_dir, dataset, phase="train", config_overrides={}):
    model_descriptions = []
    
    datasets = os.listdir(results_dir)
    dataset = next((s for s in datasets if dataset in s), None) # Find partial matches so you can use the dataset "iwildcam" for example
    algorithms = os.listdir(os.path.join(results_dir, dataset))
    for a in algorithms:
        seed_dir = os.path.join(results_dir, dataset, a, phase, dataset)
        seeds = []
        try:
            seeds = os.listdir(seed_dir)
        except Exception as e:
            print(f"Warning: Failed to read dir: {seed_dir}")
            continue

        for s in seeds:
            try:
                with open(os.path.join(seed_dir, s, "hparams.yaml"), "r") as f:
                    config = yaml.safe_load(f)["config"]
                    ckpts = os.listdir(os.path.join(seed_dir, s, "checkpoints"))
                    ckpt = next(element for element in ckpts if "best_val" in element)
                    ckpt_path = os.path.join(seed_dir, s, "checkpoints", ckpt)

                    # Do some sanity checks
                    if config["algorithm"] != a:
                        algname = config["algorithm"]
                        print(f"Warning: algorithm name {algname} does not match directory name {a}")
                    
                    # NOTE: This is to handle the adversial sweep models which got reorganized by hand
                    if config["algorithm"] != "adversarial_sweep" and config["algorithm"] != "model_sweep_adversarial":
                        assert config["algorithm"] == a

                    # Function to call to construct the model
                    def loadckpt(args={}, config=config, ckpt_path=ckpt_path):
                        config.update(args)  # Update the config with the args
                        return ClassificationTask.load_from_checkpoint(
                            checkpoint_path=ckpt_path, config=config
                        )

                    model_descriptions.append((loadckpt, s))

            except Exception as e:
                print(f"error: {e}")
                print(f"Warning: Failed to load model or config for {dataset} {a} {s}")
    return model_descriptions


def load_resnet50(
    path, out_dim, wilds_save_format=True, prefix="model.", device="cuda"
):
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, out_dim)

    state = torch.load(path, map_location=torch.device(device))

    if wilds_save_format:
        state = state["algorithm"]
        newstate = {}
        for k in state:
            if k == "model.1.weight":
                newstate["fc.weight"] = state[k]
            elif k == "model.1.bias":
                newstate["fc.bias"] = state[k]
            else:
                newstate[k.removeprefix(prefix)] = state[k]

        state = newstate

    model.load_state_dict(state)
    return model


def load_featurized_resnet50(
    path,
    out_dim,
    featurizer_prefix="model.featurizer.",
    classifier_prefix="model.classifier.",
    device="cuda",
):
    featurizer = torchvision.models.resnet50()
    featurizer_d_out = featurizer.fc.in_features
    featurizer.fc = nn.Identity(featurizer_d_out)

    classifier = torch.nn.Linear(featurizer_d_out, out_dim)

    state = torch.load(path, map_location=torch.device(device))

    state = state["algorithm"]
    featurizer_state = {}
    classifier_state = {}
    for k in state:
        if featurizer_prefix in k:
            featurizer_state[k.removeprefix(featurizer_prefix)] = state[k]
        elif classifier_prefix in k:
            classifier_state[k.removeprefix(classifier_prefix)] = state[k]

    featurizer.load_state_dict(featurizer_state)
    classifier.load_state_dict(classifier_state)
    return nn.Sequential(featurizer, classifier)


def load_densenet121(
    path, out_dim, wilds_save_format=True, prefix="model.", device="cuda"
):
    model = torchvision.models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, out_dim)

    state = torch.load(path, map_location=torch.device(device))

    if wilds_save_format:
        state = state["algorithm"]
        newstate = {}
        for k in state:
            if k == "model.1.weight":
                newstate["classifier.weight"] = state[k]
            elif k == "model.1.bias":
                newstate["classifier.bias"] = state[k]
            else:
                newstate[k.removeprefix(prefix)] = state[k]

        state = newstate

    model.load_state_dict(state)
    return model


def load_featurized_densenet121(
    path,
    out_dim,
    featurizer_prefix="featurizer.",
    classifier_prefix="classifier.",
    device="cuda",
):
    featurizer = torchvision.models.densenet121()
    featurizer_d_out = featurizer.classifier.in_features
    featurizer.classifier = nn.Identity(featurizer_d_out)

    classifier = torch.nn.Linear(featurizer_d_out, out_dim)

    state = torch.load(path, map_location=torch.device(device))

    state = state["algorithm"]
    featurizer_state = {}
    classifier_state = {}
    for k in state:
        if featurizer_prefix in k:
            featurizer_state[k.removeprefix(featurizer_prefix)] = state[k]
        elif classifier_prefix in k:
            classifier_state[k.removeprefix(classifier_prefix)] = state[k]

    featurizer.load_state_dict(featurizer_state)
    classifier.load_state_dict(classifier_state)
    return nn.Sequential(featurizer, classifier)


# Prepare model description
def hre_model_desc(model_desc, seed):
    def fn(args={}, model_desc=model_desc, seed=seed):
        config = load_config(model_desc["config"])
        config.update(args)  # Update the config with the args

        filename_fn = model_desc["filename_fn"]
        load_fn = model_desc["load_fn"]
        args = model_desc["args"]

        # Set the device
        device = "cuda" if config["accelerator"] == "gpu" else config["accelerator"]
        args["device"] = device

        # Set the algorithm name, checkpoint path and seed
        config["algorithm"] = model_desc["name"]
        config["checkpoint_path"] = filename_fn(seed)
        config["seed"] = seed

        wilds_model = load_fn(config["checkpoint_path"], config["n_classes"], **args)
        return ClassificationTask(config, wilds_model)

    return (fn, seed)


########################################################################
# Camelyon17
########################################################################
def camelyon17_pretrained_models(
    model_dir, Nseeds=-1, config=f"{thisfilepath}/../configs/camelyon17-defaults.yml"
):
    if Nseeds < 0:
        Nseeds = 10  # Set the number of available seeds

    model_descriptions = []

    model_descriptions.append(
        {
            "name": "erm",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/camelyon17_erm_densenet121_seed{s}/best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "deepCORAL",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/camelyon17_deepCORAL_densenet121_seed{s}/best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "groupDRO",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/camelyon17_groupDRO_densenet121_seed{s}/best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "irm",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/camelyon17_irm_densenet121_seed{s}/best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "AFN",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/camelyon17_afn_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "featurizer.",
                "classifier_prefix": "classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "DANN",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/camelyon17_dann_coarse_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "model.featurizer.",
                "classifier_prefix": "model.classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "deepCORAL-Coarse",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/camelyon17_deepcoral_coarse_singlepass_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "featurizer.",
                "classifier_prefix": "classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "erm-augment",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/camelyon17_ermaugment_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "erm-v2",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/camelyon17_erm_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "NoisyStudent-trainunlabeled",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/camelyon17_noisystudent_trainunlabeled_seed{s}/student1/camelyon17_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "model.featurizer.",
                "classifier_prefix": "model.classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "NoisyStudent-valunlabeled",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/camelyon17_noisystudent_valunlabeled_seed{s}/student1/camelyon17_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "model.featurizer.",
                "classifier_prefix": "model.classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "NoisyStudent-testunlabeled",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/camelyon17_noisystudent_testunlabeled_seed{s}/student1/camelyon17_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "model.featurizer.",
                "classifier_prefix": "model.classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "FixMatch",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/camelyon17_fixmatch_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {"prefix": "model.0."},
        }
    )

    model_descriptions.append(
        {
            "name": "PseudoLabels",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/camelyon17_pseudolabel_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "SwAV",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/camelyon17_swav55_ermaugment_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {"prefix": "model.0."},
        }
    )

    return [hre_model_desc(m, s) for m in model_descriptions for s in range(Nseeds)]


########################################################################
# iWildCam
########################################################################
def iwildcam_pretrained_models(
    model_dir, Nseeds=-1, config=f"{thisfilepath}/../configs/iwildcam-defaults.yml"
):
    if Nseeds < 0:
        Nseeds = 3  # Set the number of available seeds

    model_descriptions = []

    model_descriptions.append(
        {
            "name": "erm",
            "load_fn": load_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/iwildcam_erm_seed{s}/best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "deepCORAL",
            "load_fn": load_featurized_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/iwildcam_deepCORAL_seed{s}/best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "featurizer.",
                "classifier_prefix": "classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "groupDRO",
            "load_fn": load_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/iwildcam_groupDRO_seed{s}/best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "irm",
            "load_fn": load_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/iwildcam_irm_seed{s}/best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "AFN",
            "load_fn": load_featurized_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/iwildcam_afn_extraunlabeled_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "featurizer.",
                "classifier_prefix": "classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "DANN",
            "load_fn": load_featurized_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/iwildcam_dann_coarse_extraunlabeled_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "deepCORAL-Coarse",
            "load_fn": load_featurized_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/iwildcam_deepcoral_coarse_singlepass_extraunlabeled_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "featurizer.",
                "classifier_prefix": "classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "erm-augment",
            "load_fn": load_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/iwildcam_ermaugment_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "erm-v2",
            "load_fn": load_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/iwildcam_erm_seed{s}/iwildcam_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "NoisyStudent-extraunlabeled",
            "load_fn": load_featurized_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/iwildcam_noisystudent_extraunlabeled_seed{s}/student1/iwildcam_seed_{s}_epoch_best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "FixMatch",
            "load_fn": load_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/iwildcam_fixmatch_extraunlabeled_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
            "config": config,
            "args": {"prefix": "model.0."},
        }
    )

    model_descriptions.append(
        {
            "name": "PseudoLabels",
            "load_fn": load_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/iwildcam_pseudolabel_extraunlabeled_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "SwAV",
            "load_fn": load_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/iwildcam_swav30_ermaugment_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
            "config": config,
            "args": {"prefix": "model.0."},
        }
    )
    return [hre_model_desc(m, s) for m in model_descriptions for s in range(Nseeds)]


########################################################################
# FMOW
########################################################################
def fmow_pretrained_models(model_dir, Nseeds=-1, config=f"{thisfilepath}/../configs/fmow-defaults.yml"):
    if Nseeds < 0:
        Nseeds = 3  # Set the number of available seeds

    model_descriptions = []

    model_descriptions.append(
        {
            "name": "erm",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/fmow_erm_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "deepCORAL",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/fmow_deepcoral_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "groupDRO",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/fmow_groupDRO_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "irm",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/fmow_irm_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "AFN",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/fmow_afn_testunlabeled_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "featurizer.",
                "classifier_prefix": "classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "DANN",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/fmow_dann_coarse_testunlabeled_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "model.featurizer.",
                "classifier_prefix": "model.classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "deepCORAL-Coarse",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/fmow_deepcoral_coarse_singlepass_testunlabeled_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "featurizer.",
                "classifier_prefix": "classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "erm-augment",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/fmow_ermaugment_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "erm-v2",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/fmow_erm_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "NoisyStudent-trainunlabeled",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/fmow_noisystudent_trainunlabeled_seed{s}/student1/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "model.featurizer.",
                "classifier_prefix": "model.classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "NoisyStudent-valunlabeled",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/fmow_noisystudent_valunlabeled_seed{s}/student1/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "model.featurizer.",
                "classifier_prefix": "model.classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "NoisyStudent-testunlabeled",
            "load_fn": load_featurized_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/fmow_noisystudent_testunlabeled_seed{s}/student1/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "model.featurizer.",
                "classifier_prefix": "model.classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "FixMatch",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/fmow_fixmatch_testunlabeled_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {"prefix": "model.0."},
        }
    )

    model_descriptions.append(
        {
            "name": "PseudoLabels",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/fmow_pseudolabel_testunlabeled_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "SwAV",
            "load_fn": load_densenet121,
            "filename_fn": lambda s: f"{model_dir}/wilds_v2.0/fmow_swav35_ermaugment_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )
    return [hre_model_desc(m, s) for m in model_descriptions for s in range(Nseeds)]


########################################################################
# RxRx1
########################################################################
def rxrx1_pretrained_models(
    model_dir, Nseeds=-1, config=f"{thisfilepath}/../configs/rxrx1-defaults.yml"
):
    if Nseeds < 0:
        Nseeds = 3  # Set the number of available seeds

    model_descriptions = []

    model_descriptions.append(
        {
            "name": "erm",
            "load_fn": load_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/rxrx1_erm_seed{s}/rxrx1_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "deepCORAL",
            "load_fn": load_featurized_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/rxrx1_coral_seed{s}/rxrx1_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {
                "featurizer_prefix": "featurizer.",
                "classifier_prefix": "classifier.",
            },
        }
    )

    model_descriptions.append(
        {
            "name": "groupDRO",
            "load_fn": load_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/rxrx1_groupDRO_seed{s}/rxrx1_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )

    model_descriptions.append(
        {
            "name": "irm",
            "load_fn": load_resnet50,
            "filename_fn": lambda s: f"{model_dir}/wilds_v1.0/rxrx1_irm_seed{s}/rxrx1_seed:{s}_epoch:best_model.pth",
            "config": config,
            "args": {},
        }
    )
    return [hre_model_desc(m, s) for m in model_descriptions for s in range(Nseeds)]
