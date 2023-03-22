import sys, os
sys.path.append(os.path.dirname(__file__))
from load_wilds_models import *
from evaluate import evaluate_all_seeds
from utils import load_config

config = load_config("configs/iwildcam-defaults.yml")
    
model_dir = "/scratch/users/acorso/wilds_models/iwildcam"
Nseeds = 3
save_dir = "evaluation_results/iwildcam"

kwargs = {"config":config, "Nseeds":Nseeds, "results_dir":save_dir, }

evaluate_all_seeds(
    "erm",
    load_resnet50,
    lambda s: f"{model_dir}/wilds_v1.0/iwildcam_erm_seed{s}/best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "deepCORAL",
    load_featurized_resnet50,
    lambda s: f"{model_dir}/wilds_v1.0/iwildcam_deepCORAL_seed{s}/best_model.pth",
    **kwargs,
    args={'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}
)

evaluate_all_seeds(
    "groupDRO",
    load_resnet50,
    lambda s: f"{model_dir}/wilds_v1.0/iwildcam_groupDRO_seed{s}/best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "irm",
    load_resnet50,
    lambda s: f"{model_dir}/wilds_v1.0/iwildcam_irm_seed{s}/best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "AFN",
    load_featurized_resnet50,
    lambda s: f"{model_dir}/wilds_v2.0/iwildcam_afn_extraunlabeled_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
    **kwargs,
    args={'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}
)

evaluate_all_seeds(
    "DANN",
    load_featurized_resnet50,
    lambda s: f"{model_dir}/wilds_v2.0/iwildcam_dann_coarse_extraunlabeled_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "deepCORAL-Coarse",
    load_featurized_resnet50,
    lambda s: f"{model_dir}/wilds_v2.0/iwildcam_deepcoral_coarse_singlepass_extraunlabeled_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
    **kwargs,
    args={'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}
)

evaluate_all_seeds(
    "erm-augment",
    load_resnet50,
    lambda s: f"{model_dir}/wilds_v2.0/iwildcam_ermaugment_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "erm-v2",
    load_resnet50,
    lambda s: f"{model_dir}/wilds_v2.0/iwildcam_erm_seed{s}/iwildcam_seed:{s}_epoch:best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "NoisyStudent-extraunlabeled",
    load_featurized_resnet50,
    lambda s: f"{model_dir}/wilds_v2.0/iwildcam_noisystudent_extraunlabeled_seed{s}/student1/iwildcam_seed_{s}_epoch_best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "FixMatch",
    load_resnet50,
    lambda s: f"{model_dir}/wilds_v2.0/iwildcam_fixmatch_extraunlabeled_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
    **kwargs,
    args={"prefix": "model.0."},
)

evaluate_all_seeds(
    "PseudoLabels",
    load_resnet50,
    lambda s: f"{model_dir}/wilds_v2.0/iwildcam_pseudolabel_extraunlabeled_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "SwAV",
    load_resnet50,
    lambda s: f"{model_dir}/wilds_v2.0/iwildcam_swav30_ermaugment_seed{s}/iwildcam_seed_{s}_epoch_best_model.pth",
    **kwargs,
    args={"prefix": "model.0."},
)
