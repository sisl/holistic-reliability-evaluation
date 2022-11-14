import torch
import sys 
sys.path.append('/Users/anthonycorso/Workspace/holistic-reliability-evaluation/')
import pickle

from holistic_reliability_evaluation.load_datasets import load_camelyon17
from holistic_reliability_evaluation.load_models import load_densenet121
from holistic_reliability_evaluation.eval_functions import eval_accuracy, predict_model, eval_error_correlation, eval_adv_robust_accuracy, eval_ece

nbatches=1
shuffle_data=False
nexamples_adv = 1

data_dir = "data/"
device = torch.device('mps')
out_dim = 2

# Load Datasets
ID_dataset = load_camelyon17(data_dir, split="val", shuffle=shuffle_data)
OD_dataset = load_camelyon17(data_dir, split="test", shuffle=shuffle_data)

ID_dataset_random = load_camelyon17(data_dir, split="val", shuffle=True, batch_size=nexamples_adv)
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
    
    ID_advrob_acc = 0 #eval_adv_robust_accuracy(model, ID_dataset_random)
    OD_advrob_acc = 0 #eval_adv_robust_accuracy(model, OD_dataset_random)
    
    ID_ece = eval_ece(ID_true, ID_logits)
    OD_ece = eval_ece(OD_true, OD_logits)
    
    outputs[name] = {'ID_true':ID_true, 
                     'ID_logits': ID_logits, 
                     'OD_true':OD_true, 
                     'OD_logits':OD_logits}
    results[name] = {'ID_advrob_acc':ID_advrob_acc,
                     'OD_advrob_acc':OD_advrob_acc,
                     'ID_ece':ID_ece,
                     'OD_ece':OD_ece}
    
    

# pickle the results
file = open('outputs', 'wb')
pickle.dump(outputs, file)
file.close()

file = open('results', 'wb')
pickle.dump(results, file)
file.close()

# file = open('results', 'rb')
# results = pickle.load(file)
# file.close()

# Evaluate the error correlations
for name in outputs:
    ID_true, ID_logits = outputs[name]['ID_true'], outputs[name]['ID_logits']
    OD_true, OD_logits = outputs[name]['OD_true'], outputs[name]['OD_logits']

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

for name in results:    
    print(results[name])