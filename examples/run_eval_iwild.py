import torch
import pickle
import sys
import random

sys.path.append('/home/dk11/holistic-reliability-evaluation/')
sys.path.append('/home/dk11/augmentation-corruption/imagenet_c_bar')
sys.path.append('/home/dk11/augmentation-corruption/imagenet_c_bar/utils')

from holistic_reliability_evaluation.load_datasets import load_wilds_dataset, random_noise_dataset
from holistic_reliability_evaluation.load_models import load_densenet121, load_featurized_densenet121, load_resnet50, load_featurized_resnet50
from holistic_reliability_evaluation.evaluation import ModelGroup, Model, EvaluationSuite, ConformalPredictionParams

data_dir = "/home/dk11/data/"
model_dir = "/home/dk11/iwild_trained_models/"
results_dir = "/home/dk11/holistic-reliability-evaluation/results/"
device = torch.device('cuda')
torch.cuda.set_device(0)
print(torch.cuda.get_device_name(0))

out_dim = 182

evaluator = EvaluationSuite(
    { # In-distribution dataset
        "ID": load_wilds_dataset("iwildcam", data_dir, split="id_val", resize=(448,448))
    },
    { # Domain shifted datastes
        "Real World DS": load_wilds_dataset("iwildcam", data_dir, split="test", resize=(448,448)),
        "C Bar (severity 1)":load_wilds_dataset("iwildcam", data_dir, split="id_val", corruption=1, resize=(448,448)),
        "C Bar (severity 5)":load_wilds_dataset("iwildcam", data_dir, split="id_val", corruption=5, resize=(448,448))
    },
    { # OOD Datsets
        "RxRx1": load_wilds_dataset("rxrx1", data_dir, split="id_test", resize=(448,448)), 
        "Gaussian Noise": random_noise_dataset(items=35000, size=(448,448))
    },
    run_test = True,
    test_size=2048,
    adv_acc = True,
    num_adv_examples = 32,
    uq_metrics = ["ece", ConformalPredictionParams(500, 0.2, "Softmax"), ConformalPredictionParams(500, 0.2, "Acc. Softmax")],
)

print("done loading dataset")

# Load models
Nseeds = 3

erm_models = ModelGroup('ERM', Nseeds, lambda s : f"{model_dir}//wilds_v2.0/iwildcam_erm_seed{s}/best_model.pth", load_resnet50, {'out_dim':out_dim}, device, nograd_batchsize=128, grad_batchsize=16)
dro_models = ModelGroup('groupDRO', Nseeds, lambda s : f"{model_dir}/wilds_v2.0/iwildcam_groupDRO_seed{s}/best_model.pth", load_resnet50, {'out_dim':out_dim},device, nograd_batchsize=128, grad_batchsize=16)
irm_models = ModelGroup('IRM', Nseeds, lambda s : f"{model_dir}/wilds_v2.0/iwildcam_irm_seed{s}/best_model.pth", load_resnet50, {'out_dim':out_dim}, device, nograd_batchsize=128, grad_batchsize=16)
ermaugment_models = ModelGroup('ERM-Augment', Nseeds, lambda s : f"{model_dir}/wilds_unlabeled/iwildcam_ermaugment_seed{s}/iwildcam_seed:{s}_epoch:best_model.pth", load_resnet50, {'out_dim':out_dim}, device, nograd_batchsize=128, grad_batchsize=16)
swav_models = ModelGroup('SwAV', Nseeds, lambda s : f"{model_dir}/wilds_unlabeled/iwildcam_swav30_ermaugment_seed{s}/iwildcam_seed:{s}_epoch:best_model.pth", load_resnet50, {'out_dim':out_dim, 'prefix':'model.0.'}, device, nograd_batchsize=128, grad_batchsize=16)
fixmatch_models = ModelGroup('FixMatch', Nseeds, lambda s : f"{model_dir}/wilds_unlabeled/iwildcam_fixmatch_extraunlabeled_seed{s}/iwildcam_seed:{s}_epoch:best_model.pth", load_resnet50, {'out_dim':out_dim, 'prefix':'model.0.'}, device, nograd_batchsize=128, grad_batchsize=16)
pseudolabels_models = ModelGroup('PseudoLabels', Nseeds, lambda s : f"{model_dir}/wilds_unlabeled/iwildcam_pseudolabel_extraunlabeled_seed{s}/iwildcam_seed:{s}_epoch:best_model.pth", load_resnet50, {'out_dim':out_dim}, device, nograd_batchsize=128, grad_batchsize=16)
ermoracle_models = ModelGroup('ERMORACLE', Nseeds, lambda s : f"{model_dir}/wilds_unlabeled/iwildcam_ermoracle_extraunlabeled_seed{s}/iwildcam_seed:{s}_epoch:best_model.pth", load_resnet50, {'out_dim':out_dim}, device, nograd_batchsize=128, grad_batchsize=16)
noisy_student_models = ModelGroup('Noisy-Student', Nseeds, lambda s : f"{model_dir}/wilds_unlabeled/iwildcam_noisystudent_extraunlabeled_seed{s}/student1/iwildcam_seed:{s}_epoch:best_model.pth", load_featurized_resnet50, {'out_dim':out_dim}, device, nograd_batchsize=128, grad_batchsize=16)
dann_coarse_models = ModelGroup('DANN-COURSE', Nseeds, lambda s : f"{model_dir}/wilds_unlabeled/iwildcam_dann_coarse_extraunlabeled_seed{s}/iwildcam_seed:{s}_epoch:best_model.pth", load_featurized_resnet50, {'out_dim':out_dim}, device, nograd_batchsize=128, grad_batchsize=16)
coral_models = ModelGroup('deepCORAL', Nseeds, lambda s : f"{model_dir}/wilds_v2.0/iwildcam_deepCORAL_seed{s}/best_model.pth", load_featurized_resnet50, {'out_dim':out_dim, 'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}, device, nograd_batchsize=128, grad_batchsize=16)
afn_models = ModelGroup('AFN', Nseeds, lambda s : f"{model_dir}/wilds_unlabeled/iwildcam_afn_extraunlabeled_seed{s}/iwildcam_seed:{s}_epoch:best_model.pth", load_featurized_resnet50, {'out_dim':out_dim, 'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}, device, nograd_batchsize=128, grad_batchsize=16)
deepcoral_coarse_models = ModelGroup('DEEPCORAL-COURSE', Nseeds, lambda s : f"{model_dir}/wilds_unlabeled/iwildcam_deepcoral_coarse_singlepass_extraunlabeled_seed{s}/iwildcam_seed:{s}_epoch:best_model.pth", load_featurized_resnet50, {'out_dim':out_dim, 'featurizer_prefix':'featurizer.', 'classifier_prefix':'classifier.'}, device, nograd_batchsize=128, grad_batchsize=16)


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
          *ermoracle_models.gen_models()
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