import torch
import pickle
import sys

sys.path.append('/Users/anthonycorso/Workspace/augmentation-corruption/imagenet_c_bar')
sys.path.append('/Users/anthonycorso/Workspace/augmentation-corruption/imagenet_c_bar/utils')

from holistic_reliability_evaluation.load_datasets import load_wilds_dataset, random_noise_dataset
from holistic_reliability_evaluation.load_models import load_densenet121
from holistic_reliability_evaluation.evaluation import Model, EvaluationSuite

gradient_batch_size=32 # Batch sized used when also computing gradients
nograd_batch_size=1024 # Batch size used when performing inference w/o gradient
nexamples_adv=256 # Previously set to 250 for first round of experiments.

data_dir = "/Users/anthonycorso/Workspace/wilds/data/"
model_dir = "/Users/anthonycorso/Workspace/holistic-reliability-evaluation/trained_models/"
device = torch.device('cpu')
out_dim = 2

evaluator = EvaluationSuite(
    { # In-distribution dataset
        "ID": load_wilds_dataset("camelyon17", data_dir, split="id_val")
    },
    { # Domain shifted datastes
        "Real World DS": load_wilds_dataset("camelyon17", data_dir, split="test"),
        "C Bar (severity 1)":load_wilds_dataset("camelyon17", data_dir, split="id_val", corruption=1),
        "C Bar (severity 5)":load_wilds_dataset("camelyon17", data_dir, split="id_val", corruption=5)
    },
    { # OOD Datsets
        "RxRx1": load_wilds_dataset("rxrx1", data_dir, split="id_test"), 
        "Gaussian Noise": random_noise_dataset(items=35000)
    },
    run_test=True,
    adv_acc=False
)

# Load models
erm0 = load_densenet121(f"{model_dir}/camelyon17_erm_densenet121_seed0/best_model.pth", out_dim, device=device)
erm1 = load_densenet121(f"{model_dir}/camelyon17_erm_densenet121_seed1/best_model.pth", out_dim, device=device)
swav0 = load_densenet121(f"{model_dir}/camelyon17_swav55_ermaugment_seed0/camelyon17_seed:0_epoch:best_model.pth", out_dim, prefix='model.0.', device=device)
swav1 = load_densenet121(f"{model_dir}/camelyon17_swav55_ermaugment_seed1/camelyon17_seed:1_epoch:best_model.pth", out_dim, prefix='model.0.', device=device)

model1 = Model(erm0)
evaluator.evaluate(model1)

# models = [Model(erm0), Model(erm1), Model(swav0), Model(swav1)]

# for model in models:
#     evaluator.evaluate(model)


# models = [erm0, erm1, swav0, swav1]
# names = ["ERM-seed0", "ERM-seed1", "SWAV-seed0", "SWAV-seed1"]




# #file = open('results', 'rb')
# #results = pickle.load(file)
# #file.close()

# #file = open('outputs', 'rb')
# #outputs = pickle.load(file)
# #file.close()

# # Evaluate the error correlations
# for name in outputs:
#     ID_true, ID_logits = outputs[name]['ID_true'], outputs[name]['ID_logits']
#     DS_true, DS_logits = outputs[name]['DS_true'], outputs[name]['DS_logits']
    
#     ID_accuracy = eval_accuracy(ID_true, ID_logits)
#     DS_accuracy = eval_accuracy(DS_true, DS_logits)
    
#     results[name]['ID_accuracy'] = ID_accuracy
#     results[name]['DS_accuracy'] = DS_accuracy

#     ID_correlations = []
#     DS_correlations = []
#     for name2 in outputs:
#         ID_true2, ID_logits2 = outputs[name2]['ID_true'], outputs[name2]['ID_logits']
#         DS_true2, DS_logits2 = outputs[name2]['DS_true'], outputs[name2]['DS_logits']
#         ID_correlation = eval_error_correlation(ID_true, ID_logits, ID_true2, ID_logits2)
#         DS_correlation = eval_error_correlation(DS_true, DS_logits, DS_true2, DS_logits2)
#         ID_correlations.append(ID_correlation)
#         DS_correlations.append(DS_correlation)
#     results[name]['ID_correlations'] = ID_correlations
#     results[name]['DS_correlations'] = DS_correlations
    
# # pickle the results
# file = open('outputs', 'wb')
# pickle.dump(outputs, file)
# file.close()

# file = open('results', 'wb')
# pickle.dump(results, file)
# file.close()

# for name in results:     
#     print(name, " --> ", results[name])
