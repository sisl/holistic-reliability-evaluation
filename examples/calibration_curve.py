import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
import sys
sys.path.append('/Users/anthonycorso/Workspace/augmentation-corruption/imagenet_c_bar')
sys.path.append('/Users/anthonycorso/Workspace/augmentation-corruption/imagenet_c_bar/utils')


from torch.utils.data import DataLoader
from holistic_reliability_evaluation.load_datasets import load_wilds_dataset, random_noise_dataset
from holistic_reliability_evaluation.load_models import load_densenet121
import pytorch_ood as ood


# Load some models
erm = load_densenet121("trained_models/camelyon17_erm_densenet121_seed0/best_model.pth", out_dim=2, force_cpu=True)
swav = load_densenet121("trained_models/camelyon17_swav55_ermaugment_seed0/camelyon17_seed:0_epoch:best_model.pth", out_dim=2, prefix='model.0.', force_cpu=True)

data_dir = "/Users/anthonycorso/Workspace/wilds/data/"
ID_dataset_random = DataLoader(load_wilds_dataset("camelyon17", data_dir, split="id_val"), shuffle=True, batch_size=100)
DS_dataset_random = DataLoader(load_wilds_dataset("camelyon17", data_dir, split="test"), shuffle=True, batch_size=100)
val_dataset_random = DataLoader(load_wilds_dataset("camelyon17", data_dir, split="val"), shuffle=True, batch_size=100)

rxrx1 = DataLoader(load_wilds_dataset("rxrx1", data_dir, split="id_test"), shuffle=True, batch_size=100)
random_data = DataLoader(random_noise_dataset(items=350), shuffle=True, batch_size=100)
cbar5 = DataLoader(load_wilds_dataset("camelyon17", data_dir, split="id_val", corruption=5))


detector = ood.detector.MaxSoftmax(swav)
metrics = ood.utils.OODMetrics()

x, y, md = next(iter(ID_dataset_random)) 
metrics.update(detector(x), y)

x, y, md = next(iter(rxrx1))
metrics.update(detector(x), -1 * torch.ones(x.shape[0])) 

id_scores = -1*metrics.buffer["scores"][metrics.buffer['labels'] == 0]
od_scores = -1*metrics.buffer["scores"][metrics.buffer['labels'] == 1]

fig, ax = plt.subplots()
ax.hist(id_scores, alpha=0.3, linewidth=1, edgecolor='black', label="ID (Cameylon17)")
ax.hist(od_scores, alpha=0.3, linewidth=1, edgecolor='black', label="OD (RxRx1)")
ax.set_title('SWAV - OOD Detection')
ax.legend(loc="upper left")

# plt.show()
plt.savefig("OOD_scores.png")


metrics.buffer['labels']