import torch
import sys 
sys.path.append('/home/acorso/Workspace/holistic-reliability-evaluation/')
import pickle

from holistic_reliability_evaluation.load_datasets import load_wilds_dataset, random_noise_dataset
from holistic_reliability_evaluation.load_models import load_densenet121
from holistic_reliability_evaluation.eval_functions import eval_accuracy, predict_model, eval_error_correlation, eval_adv_robust_accuracy, eval_ece

nbatches=1 # Set this to None if you want full evaluation
shuffle_data=False
nexamples_adv = 1 # Previously set to 250 for first round of experiments.

data_dir = "data/"
device = torch.device('mps')
out_dim = 2

# Load Datasets
ID_dataset = load_wilds_dataset("camelyon17", data_dir, split="id_val", shuffle=shuffle_data)
DS_dataset = load_wilds_dataset("camelyon17", data_dir, split="test", shuffle=shuffle_data)
OD1_dataset = load_wilds_dataset("rxrx1", data_dir, split="test", shuffle=shuffle_data, resize=(256, 256))
OD2_dataset = random_noise_dataset()

# Datasets that are randomized (use for Adv robustness evaluation)
ID_dataset_random = load_wilds_dataset("camelyon17", data_dir, split="id_val", shuffle=True, batch_size=nexamples_adv)
DS_dataset_random = load_wilds_dataset("camelyon17", data_dir, split="test", shuffle=True, batch_size=nexamples_adv)

# Load models
erm0 = load_densenet121("trained_models/camelyon17_erm_densenet121_seed0/best_model.pth", out_dim)
erm1 = load_densenet121("trained_models/camelyon17_erm_densenet121_seed1/best_model.pth", out_dim)
swav0 = load_densenet121("trained_models/camelyon17_swav55_ermaugment_seed0/camelyon17_seed:0_epoch:best_model.pth", out_dim, prefix='model.0.')
swav1 = load_densenet121("trained_models/camelyon17_swav55_ermaugment_seed1/camelyon17_seed:1_epoch:best_model.pth", out_dim, prefix='model.0.')

models = [erm0, erm1, swav0, swav1]
names = ["ERM-seed0", "ERM-seed1", "SWAV-seed0", "SWAV-seed1"]

# Compute model outputs
results = {}
outputs = {}
for model, name in zip(models, names):
    print('solving: ', name)
    ID_true, ID_logits = predict_model(model, ID_dataset, nbatches=nbatches, device=device)
    DS_true, DS_logits = predict_model(model, DS_dataset, nbatches=nbatches, device=device)
    OD1_true, OD1_logits = predict_model(model, OD1_dataset, nbatches=nbatches, device=device)
    OD2_true, OD2_logits = predict_model(model, OD2_dataset, nbatches=nbatches, device=device)
    
    
    ID_accuracy = eval_accuracy(ID_true, ID_logits)
    DS_accuracy = eval_accuracy(DS_true, DS_logits)
    
    ID_advrob_acc = eval_adv_robust_accuracy(model, ID_dataset_random, device=device)
    DS_advrob_acc = eval_adv_robust_accuracy(model, DS_dataset_random, device=device)
    
    ID_ece = eval_ece(ID_true, ID_logits)
    DS_ece = eval_ece(DS_true, DS_logits)
    
    outputs[name] = {'ID_true':ID_true, 
                     'ID_logits': ID_logits, 
                     'DS_true':DS_true, 
                     'DS_logits':DS_logits,
                     'OD1_true':OD1_true, 
                     'OD1_logits':OD1_logits,
                     'OD2_true':OD2_true, 
                     'OD2_logits':OD2_logits}
    results[name] = {'ID_accuracy':ID_accuracy,
                     'DS_accuracy':DS_accuracy,
                     'ID_advrob_acc':ID_advrob_acc,
                     'DS_advrob_acc':DS_advrob_acc,
                     'ID_ece':ID_ece,
                     'DS_ece':DS_ece}

#file = open('results', 'rb')
#results = pickle.load(file)
#file.close()

#file = open('outputs', 'rb')
#outputs = pickle.load(file)
#file.close()

# Evaluate the error correlations
for name in outputs:
    ID_true, ID_logits = outputs[name]['ID_true'], outputs[name]['ID_logits']
    DS_true, DS_logits = outputs[name]['DS_true'], outputs[name]['DS_logits']
    
    ID_accuracy = eval_accuracy(ID_true, ID_logits)
    DS_accuracy = eval_accuracy(DS_true, DS_logits)
    
    results[name]['ID_accuracy'] = ID_accuracy
    results[name]['DS_accuracy'] = DS_accuracy

    ID_correlations = []
    DS_correlations = []
    for name2 in outputs:
        ID_true2, ID_logits2 = outputs[name2]['ID_true'], outputs[name2]['ID_logits']
        DS_true2, DS_logits2 = outputs[name2]['DS_true'], outputs[name2]['DS_logits']
        ID_correlation = eval_error_correlation(ID_true, ID_logits, ID_true2, ID_logits2)
        DS_correlation = eval_error_correlation(DS_true, DS_logits, DS_true2, DS_logits2)
        ID_correlations.append(ID_correlation)
        DS_correlations.append(DS_correlation)
    results[name]['ID_correlations'] = ID_correlations
    results[name]['DS_correlations'] = DS_correlations
    
# pickle the results
file = open('outputs', 'wb')
pickle.dump(outputs, file)
file.close()

file = open('results', 'wb')
pickle.dump(results, file)
file.close()

for name in results:     
    print(name, " --> ", results[name])
