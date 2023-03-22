import sys, os
sys.path.append(os.path.dirname(__file__))
from load_wilds_models import *
from evaluate import evaluate_all_seeds
from utils import load_config

config = load_config("configs/fmow-defaults.yml")

model_dir = "/scratch/users/acorso/wilds_models/fmow"
Nseeds = 3
save_dir = "evaluation_results/fmow"

kwargs = {"config":config, "Nseeds":Nseeds, "results_dir":save_dir, }

# evaluate_all_seeds(
#     "erm",
#     load_densenet121,
#     lambda s: f"{model_dir}/wilds_v1.0/fmow_erm_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
# )

# evaluate_all_seeds(
#     "deepCORAL",
#     load_featurized_densenet121,
#     lambda s: f"{model_dir}/wilds_v1.0/fmow_deepcoral_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
# )

# evaluate_all_seeds(
#     "groupDRO",
#     load_densenet121,
#     lambda s: f"{model_dir}/wilds_v1.0/fmow_groupDRO_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
# )

# evaluate_all_seeds(
#     "irm",
#     load_densenet121,
#     lambda s: f"{model_dir}/wilds_v1.0/fmow_irm_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
# )

# evaluate_all_seeds(
#     "AFN",
#     load_featurized_densenet121,
#     lambda s: f"{model_dir}/wilds_v2.0/fmow_afn_testunlabeled_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
#     args={'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}
# )

# evaluate_all_seeds(
#     "DANN",
#     load_featurized_densenet121,
#     lambda s: f"{model_dir}/wilds_v2.0/fmow_dann_coarse_testunlabeled_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
#     args={'featurizer_prefix':'model.featurizer.', 'classifier_prefix':'model.classifier.'}
# )

# evaluate_all_seeds(
#     "deepCORAL-Coarse",
#     load_featurized_densenet121,
#     lambda s: f"{model_dir}/wilds_v2.0/fmow_deepcoral_coarse_singlepass_testunlabeled_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
#     args={'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}
# )

# evaluate_all_seeds(
#     "erm-augment",
#     load_densenet121,
#     lambda s: f"{model_dir}/wilds_v2.0/fmow_ermaugment_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
# )

# evaluate_all_seeds(
#     "erm-v2",
#     load_densenet121,
#     lambda s: f"{model_dir}/wilds_v2.0/fmow_erm_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
# )

# evaluate_all_seeds(
#     "NoisyStudent-trainunlabeled",
#     load_featurized_densenet121,
#     lambda s: f"{model_dir}/wilds_v2.0/fmow_noisystudent_trainunlabeled_seed{s}/student1/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
#     args={'featurizer_prefix':'model.featurizer.', 'classifier_prefix':'model.classifier.'}
# )

# evaluate_all_seeds(
#     "NoisyStudent-valunlabeled",
#     load_featurized_densenet121,
#     lambda s: f"{model_dir}/wilds_v2.0/fmow_noisystudent_valunlabeled_seed{s}/student1/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
#     args={'featurizer_prefix':'model.featurizer.', 'classifier_prefix':'model.classifier.'}
# )

# evaluate_all_seeds(
#     "NoisyStudent-testunlabeled",
#     load_featurized_densenet121,
#     lambda s: f"{model_dir}/wilds_v2.0/fmow_noisystudent_testunlabeled_seed{s}/student1/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
#     args={'featurizer_prefix':'model.featurizer.', 'classifier_prefix':'model.classifier.'}
# )

# evaluate_all_seeds(
#     "FixMatch",
#     load_densenet121,
#     lambda s: f"{model_dir}/wilds_v2.0/fmow_fixmatch_testunlabeled_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
#     args={"prefix": "model.0."},
# )

# evaluate_all_seeds(
#     "PseudoLabels",
#     load_densenet121,
#     lambda s: f"{model_dir}/wilds_v2.0/fmow_pseudolabel_testunlabeled_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
#     **kwargs,
# )

evaluate_all_seeds(
    "SwAV",
    load_densenet121,
    lambda s: f"{model_dir}/wilds_v2.0/fmow_swav35_ermaugment_seed{s}/fmow_seed:{s}_epoch:best_model.pth",
    **kwargs,
)
