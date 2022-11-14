import torch
import sys 
sys.path.append('/Users/anthonycorso/Workspace/holistic-reliability-evaluation/')
import pickle

from holistic_reliability_evaluation.load_datasets import load_camelyon17
from holistic_reliability_evaluation.load_models import load_densenet121
from holistic_reliability_evaluation.eval_functions import eval_accuracy, get_vals, eval_error_correlation

nbatches=None
shuffle_data=False

data_dir = "data/"
device = torch.device('mps')
out_dim = 2

# Load Datasets
ID_dataset = load_camelyon17(data_dir, split="val", shuffle=shuffle_data)
OD_dataset = load_camelyon17(data_dir, split="test", shuffle=shuffle_data)

# Load models
erm0 = load_densenet121("trained_models/camelyon17_erm_densenet121_seed0/best_model.pth", out_dim)
erm1 = load_densenet121("trained_models/camelyon17_erm_densenet121_seed1/best_model.pth", out_dim)
swav0 = load_densenet121("trained_models/camelyon17_swav55_ermaugment_seed0/camelyon17_seed:0_epoch:best_model.pth", out_dim, prefix='model.0.')
swav1 = load_densenet121("trained_models/camelyon17_swav55_ermaugment_seed1/camelyon17_seed:1_epoch:best_model.pth", out_dim, prefix='model.0.')

models = [erm0, erm1, swav0, swav1]
names = ["ERM-seed0", "ERM-seed1", "SWAV-seed0", "SWAV-seed1"]

# Compute model outputs
results = {}
for model, name in zip(models, names):
    ID_true, ID_pred = get_vals(model, ID_dataset, nbatches=nbatches, device=device)
    OD_true, OD_pred = get_vals(model, OD_dataset, nbatches=nbatches, device=device)
    
    results[name] = {'ID_true': ID_true, 'ID_pred': ID_pred, 'OD_true':OD_true, 'OD_pred':OD_pred}


# pickle the results
file = open('results', 'wb')
pickle.dump(results, file)
file.close()

# file = open('results', 'rb')
# results = pickle.load(file)
# file.close()

# Evaluate the results
for name in results:
    ID_true, ID_pred = results[name]['ID_true'], results[name]['ID_pred']
    OD_true, OD_pred = results[name]['OD_true'], results[name]['OD_pred']
    ID_acc = eval_accuracy(ID_true, ID_pred)
    OD_acc = eval_accuracy(OD_true, OD_pred)

    ID_correlations = []
    OD_correlations = []
    for name2 in results:
        ID_true2, ID_pred2 = results[name2]['ID_true'], results[name2]['ID_pred']
        OD_true2, OD_pred2 = results[name2]['OD_true'], results[name2]['OD_pred']
        ID_correlation = eval_error_correlation(ID_true, ID_pred, ID_true2, ID_pred2)
        OD_correlation = eval_error_correlation(OD_true, OD_pred, OD_true2, OD_pred2)
        ID_correlations.append(ID_correlation)
        OD_correlations.append(OD_correlation)

    print(name, " Results --> ID Acc: ", ID_acc, " OD acc: ", OD_acc, " ID correlations: ", ID_correlations, " OD_correlations: ", OD_correlations)