import sys, os
sys.path.append(os.path.dirname(__file__))
from load_wilds_models import *
from evaluate import evaluate_all_seeds
from utils import load_config

config = load_config("configs/rxrx1-defaults.yml")
    
model_dir = "/scratch/users/acorso/wilds_models/rxrx1"
Nseeds = 3
save_dir = "evaluation_results/rxrx1"

kwargs = {"config":config, "Nseeds":Nseeds, "results_dir":save_dir, }

evaluate_all_seeds(
    "erm",
    load_resnet50,
    lambda s: f"{model_dir}/wilds_v1.0/rxrx1_erm_seed{s}/rxrx1_seed:{s}_epoch:best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "deepCORAL",
    load_featurized_resnet50,
    lambda s: f"{model_dir}/wilds_v1.0/rxrx1_coral_seed{s}/rxrx1_seed:{s}_epoch:best_model.pth",
    **kwargs,
    args={'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}
)

evaluate_all_seeds(
    "groupDRO",
    load_resnet50,
    lambda s: f"{model_dir}/wilds_v1.0/rxrx1_groupDRO_seed{s}/rxrx1_seed:{s}_epoch:best_model.pth",
    **kwargs,
)

evaluate_all_seeds(
    "irm",
    load_resnet50,
    lambda s: f"{model_dir}/wilds_v1.0/rxrx1_irm_seed{s}/rxrx1_seed:{s}_epoch:best_model.pth",
    **kwargs,
)
