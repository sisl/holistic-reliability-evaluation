import sys, os
sys.path.append(os.path.dirname(__file__))
from load_wilds_models import *
from evaluate import evaluate_all_seeds
from utils import load_config

config = load_config("configs/camelyon17-defaults.yml")
model_dir = "/scratch/users/acorso/wilds_models/camelyon17"
Nseeds = 10
save_dir = "evaluation_results/camelyon17"

kwargs = {"config":config, "Nseeds":Nseeds, "results_dir":save_dir, }

evaluate_all_seeds(
    "erm",
    load_densenet121,
    lambda s: f"{model_dir}/wilds_v1.0/camelyon17_erm_densenet121_seed{s}/best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "deepCORAL",
    load_featurized_densenet121,
    lambda s: f"{model_dir}/wilds_v1.0/camelyon17_deepCORAL_densenet121_seed{s}/best_model.pth",
    **kwargs,
)
evaluate_all_seeds(
    "groupDRO",
    load_densenet121,
    lambda s: f"{model_dir}/wilds_v1.0/camelyon17_groupDRO_densenet121_seed{s}/best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "irm",
    load_densenet121,
    lambda s: f"{model_dir}/wilds_v1.0/camelyon17_irm_densenet121_seed{s}/best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "AFN",
    load_featurized_densenet121,
    lambda s: f"{model_dir}/wilds_v2.0/camelyon17_afn_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
    **kwargs,
    args={'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}
)

evaluate_all_seeds(
    "DANN",
    load_featurized_densenet121,
    lambda s: f"{model_dir}/wilds_v2.0/camelyon17_dann_coarse_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
    **kwargs,
    args={'featurizer_prefix':'model.featurizer.', 'classifier_prefix':'model.classifier.'}
)

evaluate_all_seeds(
    "deepCORAL-Coarse",
    load_featurized_densenet121,
    lambda s: f"{model_dir}/wilds_v2.0/camelyon17_deepcoral_coarse_singlepass_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
    **kwargs,
    args={'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}
)

evaluate_all_seeds(
    "erm-augment",
    load_densenet121,
    lambda s: f"{model_dir}/wilds_v2.0/camelyon17_ermaugment_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "erm-v2",
    load_densenet121,
    lambda s: f"{model_dir}/wilds_v2.0/camelyon17_erm_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "NoisyStudent-trainunlabeled",
    load_featurized_densenet121,
    lambda s: f"{model_dir}/wilds_v2.0/camelyon17_noisystudent_trainunlabeled_seed{s}/student1/camelyon17_seed:{s}_epoch:best_model.pth",
    **kwargs,
    args={'featurizer_prefix':'model.featurizer.', 'classifier_prefix':'model.classifier.'}
)

evaluate_all_seeds(
    "NoisyStudent-valunlabeled",
    load_featurized_densenet121,
    lambda s: f"{model_dir}/wilds_v2.0/camelyon17_noisystudent_valunlabeled_seed{s}/student1/camelyon17_seed:{s}_epoch:best_model.pth",
    **kwargs,
    args={'featurizer_prefix':'model.featurizer.', 'classifier_prefix':'model.classifier.'}
)

evaluate_all_seeds(
    "NoisyStudent-testunlabeled",
    load_featurized_densenet121,
    lambda s: f"{model_dir}/wilds_v2.0/camelyon17_noisystudent_testunlabeled_seed{s}/student1/camelyon17_seed:{s}_epoch:best_model.pth",
    **kwargs,
    args={'featurizer_prefix':'model.featurizer.', 'classifier_prefix':'model.classifier.'}
)

evaluate_all_seeds(
    "FixMatch",
    load_densenet121,
    lambda s: f"{model_dir}/wilds_v2.0/camelyon17_fixmatch_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
    **kwargs,
    args={"prefix": "model.0."},
)

evaluate_all_seeds(
    "PseudoLabels",
    load_densenet121,
    lambda s: f"{model_dir}/wilds_v2.0/camelyon17_pseudolabel_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "SwAV",
    load_densenet121,
    lambda s: f"{model_dir}/wilds_v2.0/camelyon17_swav55_ermaugment_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth",
    **kwargs,
    args={"prefix": "model.0."},
)
