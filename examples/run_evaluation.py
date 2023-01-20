import torch
import pickle
import sys
import random

# sys.path.append('/home/anthonycorso/Workspace/augmentation-corruption/imagenet_c_bar')
# sys.path.append('/home/anthonycorso/Workspace/augmentation-corruption/imagenet_c_bar/utils')

sys.path.append('/home/dk11/holistic-reliability-evaluation/')
sys.path.append('/home/dk11/augmentation-corruption/imagenet_c_bar')
sys.path.append('/home/dk11/augmentation-corruption/imagenet_c_bar/utils')

from holistic_reliability_evaluation.load_datasets import load_wilds_dataset, random_noise_dataset
from holistic_reliability_evaluation.load_models import load_densenet121, load_featurized_densenet121
from holistic_reliability_evaluation.evaluation import ModelGroup, Model, EvaluationSuite, ConformalPredictionParams

# data_dir = "/home/anthonycorso/Workspace/wilds/data/"
# model_dir = "/home/anthonycorso/Workspace/wilds/trained_models/"
# results_dir = "results/"
# device = torch.device('cuda')

data_dir = "/home/dk11/data/"
model_dir = "/home/dk11/camelyon_trained_models/"
results_dir = "/home/dk11/holistic-reliability-evaluation/results/"
device = torch.device('cuda')
torch.cuda.set_device(0)
print(torch.cuda.get_device_name(0))
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
    run_test = True,
    test_size=2048,
    adv_acc = True,
    num_adv_examples = 32,
    uq_metrics = ["ece", ConformalPredictionParams(500, 0.2, "Softmax"), ConformalPredictionParams(500, 0.2, "Acc. Softmax")],
)

# Load models
Nseeds = 10
erm_models = ModelGroup('ERM', Nseeds, lambda s : f"{model_dir}/wilds_v1.0/camelyon17_erm_densenet121_seed{s}/best_model.pth", load_densenet121, {'out_dim':out_dim}, device)
ermaugment_models = ModelGroup('ERM-Augment', Nseeds, lambda s : f"{model_dir}/wilds_v2.0/camelyon17_ermaugment_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth", load_densenet121, {'out_dim':out_dim}, device)
coral_models = ModelGroup('deepCORAL', Nseeds, lambda s : f"{model_dir}/wilds_v1.0/camelyon17_deepCORAL_densenet121_seed{s}/best_model.pth", load_featurized_densenet121, {'out_dim':out_dim}, device)
dro_models = ModelGroup('groupDRO', Nseeds, lambda s : f"{model_dir}/wilds_v1.0/camelyon17_groupDRO_densenet121_seed{s}/best_model.pth", load_densenet121, {'out_dim':out_dim},device)
irm_models = ModelGroup('IRM', Nseeds, lambda s : f"{model_dir}/wilds_v1.0/camelyon17_irm_densenet121_seed{s}/best_model.pth", load_densenet121, {'out_dim':out_dim}, device)
swav_models = ModelGroup('SwAV', Nseeds, lambda s : f"{model_dir}/wilds_v2.0/camelyon17_swav55_ermaugment_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth", load_densenet121, {'out_dim':out_dim, 'prefix':'model.0.'}, device)
fixmatch_models = ModelGroup('FixMatch', Nseeds, lambda s : f"{model_dir}/wilds_v2.0/camelyon17_fixmatch_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth", load_densenet121, {'out_dim':out_dim, 'prefix':'model.0.'}, device)
pseudolabels_models = ModelGroup('PseudoLabels', Nseeds, lambda s : f"{model_dir}/wilds_v2.0/camelyon17_pseudolabel_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth", load_densenet121, {'out_dim':out_dim}, device)


# Extra Models
afn_models = ModelGroup('AFN', Nseeds, lambda s : f"{model_dir}/camelyon_extra_unlabeled/camelyon17_afn_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth", load_featurized_densenet121, {'out_dim':out_dim, 'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}, device, nograd_batchsize=128, grad_batchsize=16)
deepcoral_coarse_models = ModelGroup('DEEPCORAL-COURSE', Nseeds, lambda s : f"{model_dir}/camelyon_extra_unlabeled/camelyon17_deepcoral_coarse_singlepass_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth", load_featurized_densenet121, {'out_dim':out_dim, 'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}, device, nograd_batchsize=128, grad_batchsize=16)
noisy_student_models = ModelGroup('Noisy-Student', Nseeds, lambda s : f"{model_dir}/camelyon_extra_unlabeled/camelyon17_noisystudent_testunlabeled_seed{s}/student1/camelyon17_seed:{s}_epoch:best_model.pth", load_featurized_densenet121, {'out_dim':out_dim, 'featurizer_prefix':'model.featurizer.', 'classifier_prefix':'model.classifier.'}, device, nograd_batchsize=128, grad_batchsize=16)
dann_coarse_models = ModelGroup('DANN-COURSE', Nseeds, lambda s : f"{model_dir}/camelyon_extra_unlabeled/camelyon17_dann_coarse_testunlabeled_seed{s}/camelyon17_seed:{s}_epoch:best_model.pth", load_featurized_densenet121, {'out_dim':out_dim, 'featurizer_prefix':'model.featurizer.', 'classifier_prefix':'model.classifier.'}, device, nograd_batchsize=128, grad_batchsize=16)
 
print("Done loading models...")

models = [
          *erm_models.gen_models(), 
          *ermaugment_models.gen_models(), 
          *coral_models.gen_models(),
          *dro_models.gen_models(),
          *irm_models.gen_models(),
          *swav_models.gen_models(),
          *fixmatch_models.gen_models(),
          *pseudolabels_models.gen_models(),
          *noisy_student_models.gen_models(),
          *afn_models.gen_models(),
          *dann_coarse_models.gen_models(),
          *deepcoral_coarse_models.gen_models(),
          ]

random.shuffle(models)
for model in models:
    #try:
        evaluator.evaluate(model)
        file = open(f'results/{model.name()}.pkl', 'wb')
        pickle.dump(model, file)
        file.close()
    #except Exception as e:
        #print("===============>>>>  Failed!", e)
