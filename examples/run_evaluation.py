import torch
import pickle
import sys
import random

sys.path.append('/home/anthonycorso/Workspace/augmentation-corruption/imagenet_c_bar')
sys.path.append('/home/anthonycorso/Workspace/augmentation-corruption/imagenet_c_bar/utils')

from holistic_reliability_evaluation.load_datasets import load_wilds_dataset, random_noise_dataset
from holistic_reliability_evaluation.load_models import load_densenet121, load_featurized_densenet121
from holistic_reliability_evaluation.evaluation import ModelGroup, Model, EvaluationSuite

data_dir = "/home/anthonycorso/Workspace/wilds/data/"
model_dir = "/home/anthonycorso/Workspace/wilds/trained_models/"
results_dir = "results/"
device = torch.device('cuda')
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
    run_test = False,
    test_size = 2048,
)

# Load models
Nseeds = 10
erm_models = ModelGroup('ERM', Nseeds, lambda s : f"{model_dir}/wilds_v1.0/camelyon17_erm_densenet121_seed{s}/best_model.pth", load_densenet121, {'out_dim':out_dim}, device)
ermaugment_models = ModelGroup('ERM-Augment', Nseeds, lambda s : f"{model_dir}/wilds_v2.0/camelyon17_ermaugment_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth", load_densenet121, {'out_dim':out_dim}, device)
coral_models = ModelGroup('deepCORAL', Nseeds, lambda s : f"{model_dir}/wilds_v1.0/camelyon17_deepCORAL_densenet121_seed{s}/best_model.pth", load_featurized_densenet121, {'out_dim':out_dim}, device)
dro_models = ModelGroup('groupDRO', Nseeds, lambda s : f"{model_dir}/wilds_v1.0/camelyon17_groupDRO_densenet121_seed{s}/best_model.pth", load_densenet121, {'out_dim':out_dim},device)
irm_models = ModelGroup('IRM', Nseeds, lambda s : f"{model_dir}/wilds_v1.0/camelyon17_irm_densenet121_seed{s}/best_model.pth", load_densenet121, {'out_dim':out_dim}, device)
swav_models = ModelGroup('SwAV', 2, lambda s : f"{model_dir}/wilds_v2.0/camelyon17_swav55_ermaugment_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth", load_densenet121, {'out_dim':out_dim, 'prefix':'model.0.'}, device)
fixmatch_models = ModelGroup('FixMatch', 2, lambda s : f"{model_dir}/wilds_v2.0/camelyon17_fixmatch_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth", load_densenet121, {'out_dim':out_dim, 'prefix':'model.0.'}, device)
pseudolabels_models = ModelGroup('PseudoLabels', 2, lambda s : f"{model_dir}/wilds_v2.0/camelyon17_pseudolabel_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth", load_densenet121, {'out_dim':out_dim}, device)


models = [*erm_models.gen_models(), 
          *ermaugment_models.gen_models(), 
          *coral_models.gen_models(),
          *dro_models.gen_models(),
          *irm_models.gen_models(),
          *swav_models.gen_models(),
          *fixmatch_models.gen_models(),
          *pseudolabels_models.gen_models()]

random.shuffle(models)
for model in models:
    try:
        evaluator.evaluate(model)
        file = open(f'results/{model.name()}.pkl', 'wb')
        pickle.dump(model, file)
        file.close()
    except Exception as e:
        print("===============>>>>  Failed!", e)


# models = [erm0, erm1, swav0, swav1]
# names = ["ERM-seed0", "ERM-seed1", "SWAV-seed0", "SWAV-seed1"]


# #file = open('results', 'rb')
# #results = pickle.load(file)
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
