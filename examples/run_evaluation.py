import torch
import sys 
sys.path.append('/home/acorso/Workspace/holistic-reliability-evaluation/')
import pickle

from holistic_reliability_evaluation.load_datasets import load_camelyon17
from holistic_reliability_evaluation.load_models import load_densenet121
from holistic_reliability_evaluation.eval_functions import eval_accuracy, predict_model, eval_error_correlation, eval_adv_robust_accuracy, eval_ece

nbatches=1 # Set this to None if you want full evaluation
shuffle_data=False
nexamples_adv = 1 # Previously set to 250 for first round of experiments.

data_dir = "data/"
device = torch.device('cuda')
out_dim = 2

# Load Datasets
ID_dataset = load_camelyon17(data_dir, split="id_val", shuffle=shuffle_data)
OD_dataset = load_camelyon17(data_dir, split="test", shuffle=shuffle_data)

ID_dataset_random = load_camelyon17(data_dir, split="id_val", shuffle=True, batch_size=nexamples_adv)
OD_dataset_random = load_camelyon17(data_dir, split="test", shuffle=True, batch_size=nexamples_adv)

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
    OD_true, OD_logits = predict_model(model, OD_dataset, nbatches=nbatches, device=device)
    
    ID_accuracy = eval_accuracy(ID_true, ID_logits)
    OD_accuracy = eval_accuracy(OD_true, OD_logits)
    
    ID_advrob_acc = eval_adv_robust_accuracy(model, ID_dataset_random, device=device)
    OD_advrob_acc = eval_adv_robust_accuracy(model, OD_dataset_random, device=device)
    
    ID_ece = eval_ece(ID_true, ID_logits)
    OD_ece = eval_ece(OD_true, OD_logits)
    
    outputs[name] = {'ID_true':ID_true, 
                     'ID_logits': ID_logits, 
                     'OD_true':OD_true, 
                     'OD_logits':OD_logits}
    results[name] = {'ID_accuracy':ID_accuracy,
                     'OD_accuracy':OD_accuracy,
                     'ID_advrob_acc':ID_advrob_acc,
                     'OD_advrob_acc':OD_advrob_acc,
                     'ID_ece':ID_ece,
                     'OD_ece':OD_ece}

#file = open('results', 'rb')
#results = pickle.load(file)
#file.close()

#file = open('outputs', 'rb')
#outputs = pickle.load(file)
#file.close()

# Evaluate the error correlations
for name in outputs:
    ID_true, ID_logits = outputs[name]['ID_true'], outputs[name]['ID_logits']
    OD_true, OD_logits = outputs[name]['OD_true'], outputs[name]['OD_logits']
    
    ID_accuracy = eval_accuracy(ID_true, ID_logits)
    OD_accuracy = eval_accuracy(OD_true, OD_logits)
    
    results[name]['ID_accuracy'] = ID_accuracy
    results[name]['OD_accuracy'] = OD_accuracy

    ID_correlations = []
    OD_correlations = []
    for name2 in outputs:
        ID_true2, ID_logits2 = outputs[name2]['ID_true'], outputs[name2]['ID_logits']
        OD_true2, OD_logits2 = outputs[name2]['OD_true'], outputs[name2]['OD_logits']
        ID_correlation = eval_error_correlation(ID_true, ID_logits, ID_true2, ID_logits2)
        OD_correlation = eval_error_correlation(OD_true, OD_logits, OD_true2, OD_logits2)
        ID_correlations.append(ID_correlation)
        OD_correlations.append(OD_correlation)
    results[name]['ID_correlations'] = ID_correlations
    results[name]['OD_correlations'] = OD_correlations
    
# pickle the results
file = open('outputs', 'wb')
pickle.dump(outputs, file)
file.close()

file = open('results', 'wb')
pickle.dump(results, file)
file.close()

for name in results:     
    print(name, " --> ", results[name])
