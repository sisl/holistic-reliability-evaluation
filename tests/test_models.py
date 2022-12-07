import torch
import sys
sys.path.append('/home/anthonycorso/Workspace/augmentation-corruption/imagenet_c_bar')
sys.path.append('/home/anthonycorso/Workspace/augmentation-corruption/imagenet_c_bar/utils')


from holistic_reliability_evaluation.load_datasets import load_wilds_dataset
from holistic_reliability_evaluation.load_models import load_densenet121, load_featurized_densenet121
from holistic_reliability_evaluation.evaluation import ModelGroup, Model, EvaluationSuite, get_errors

data_dir = "/home/anthonycorso/Workspace/wilds/data/"
model_dir = "/home/anthonycorso/Workspace/wilds/trained_models/"
device = torch.device('cuda')
out_dim = 2

dataset = torch.utils.data.DataLoader(load_wilds_dataset("camelyon17", data_dir, split="id_val"), shuffle=True, batch_size=100)

def quickeval(model, x, y):
    with torch.no_grad():
        pred = model(x)
    return 1 - sum(get_errors(y, pred)).item() / pred.size(0)


x, y, md = next(iter(dataset))

# Try out models
erm0 = load_densenet121(f"{model_dir}/wilds_v1.0/camelyon17_erm_densenet121_seed0/best_model.pth", out_dim)
quickeval(erm0, x, y)

deepCORAL0 = load_featurized_densenet121(f"{model_dir}/wilds_v1.0/camelyon17_deepCORAL_densenet121_seed0/best_model.pth", out_dim)
quickeval(deepCORAL0, x, y)

groupDRO0 = load_densenet121(f"{model_dir}/wilds_v1.0/camelyon17_groupDRO_densenet121_seed0/best_model.pth", out_dim)
quickeval(groupDRO0, x, y)

IRM0 = load_densenet121(f"{model_dir}/wilds_v1.0/camelyon17_irm_densenet121_seed0/best_model.pth", out_dim)
quickeval(IRM0, x, y)

swav0 = load_densenet121(f"{model_dir}/wilds_v2.0/camelyon17_swav55_ermaugment_seed0/camelyon17_seed:0_epoch:best_model.pth", out_dim, prefix='model.0.')
quickeval(swav0, x, y)

ermaugment0 = load_densenet121(f"{model_dir}/wilds_v2.0/camelyon17_ermaugment_seed0/camelyon17_seed:0_epoch:best_model.pth", out_dim)
quickeval(ermaugment0, x, y)

fixmatch0 = load_densenet121(f"{model_dir}/wilds_v2.0/camelyon17_fixmatch_testunlabeled_seed0/camelyon17_seed:0_epoch:best_model.pth", out_dim, prefix='model.0.')
quickeval(fixmatch0, x, y)

pseudolabel0 = load_densenet121(f"{model_dir}/wilds_v2.0/camelyon17_pseudolabel_testunlabeled_seed0/camelyon17_seed:0_epoch:best_model.pth", out_dim)
quickeval(pseudolabel0, x, y)


