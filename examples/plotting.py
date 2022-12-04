import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np

# Process the results of a full run
file = open('results', 'rb')
results = pickle.load(file)
file.close()

for name in results:
    print(name, " --> ", results[name])


metrics = ['ID_accuracy', 'DS_accuracy', 'ID_advrob_acc', 'DS_advrob_acc', 'ID_ece', 'DS_ece', ]
erm_models = ['ERM-seed0', 'ERM-seed1']
swav_models = ['SWAV-seed0', 'SWAV-seed1']

models_list = [erm_models, swav_models]
model_names = ["ERM", "SWAV"]

## Accuracy and robustness metrics
for models, name in zip(models_list, model_names):
    print(name, end="")
    for m in metrics:
        t = torch.tensor([results[model][m] for model in models])
        print(" & ", round(t.mean().item(), 3), " (" + str(round(t.std().item(), 3)) + ")", end="")
    print("\\\\")

## OOD detection metrics
ood_approaches = ["max_softmax", "energy_based"]
ood_approaches_names = ["Max Softmax", "Energy-Based"]
ood_datasets = ["ood_detection1", "ood_detection2"]
ood_metrics = ["AUROC", "AUPR-IN", "AUPR-OUT", "ACC95TPR", "FPR95TPR"]
for models, name in zip(models_list, model_names):
    for approach, approach_name in zip(ood_approaches, ood_approaches_names):
        print(name, " & ", approach_name, end="")
        for dataset in ood_datasets:
            key = approach + "_" + dataset
            for ood_metric in ood_metrics:
                t = torch.tensor([results[model][key][ood_metric] for model in models])
                print(" & ", round(t.mean().item(), 3), " (" + str(round(t.std().item(), 3)) + ")", end="")
        print("\\\\")

    
ID_error_correlations = torch.stack([torch.tensor(results[m]['ID_correlations']) for m in erm_models + swav_models])
OD_error_correlations = torch.stack([torch.tensor(results[m]['OD_correlations']) for m in erm_models + swav_models])

labels = erm_models + swav_models

def plot_error_correlation(error_correlations, labels, title, savefile):
    fig, ax = plt.subplots()
    im = ax.imshow(error_correlations, vmin=0, vmax=1)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(savefile)

plot_error_correlation(ID_error_correlations, labels, "ID Error Correlation", "ID_error_correlation.png")
plot_error_correlation(OD_error_correlations, labels, "OD Error Correlation", "OD_error_correlation.png")


# Produce random examples from Camelyon dataset
from holistic_reliability_evaluation.load_datasets import load_camelyon17
data_dir = "data/"
ID_dataset_random = load_camelyon17(data_dir, split="id_val", shuffle=True, batch_size=100)
OD_dataset_random = load_camelyon17(data_dir, split="test", shuffle=True, batch_size=100)
val_dataset_random = load_camelyon17(data_dir, split="val", shuffle=True, batch_size=100)

def get_random(dataset, yval, hospital):
    for i in range(10):
        x, y, md = next(iter(dataset))
        indices = (y == yval).logical_and(md[:, 0] == hospital)
        if not indices.any():
            continue
        else:
            assert((y[indices] == yval).all())
            assert((md[indices, 0] == hospital).all())
            return x[indices,:,:,:][0, :, :, :]
        
x = get_random(OD_dataset_random, 1, 2)
plt.imshow(x.permute([2, 1, 0]), interpolation='bilinear')
plt.show()


fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(8, 2),subplot_kw={'xticks': [], 'yticks': []})

datasets = [ID_dataset_random,val_dataset_random,OD_dataset_random, ID_dataset_random, ID_dataset_random,ID_dataset_random,val_dataset_random,OD_dataset_random, ID_dataset_random, ID_dataset_random]
yvals = [0,0,0,0,0,1,1,1,1,1]
hospitals = [0,1,2,3,4,0,1,2,3,4]

for ax, dataset, hospital, yval in zip(axs.flat, datasets, hospitals, yvals):
    x = get_random(dataset, yval, hospital)
    ax.imshow(x.permute([2, 1, 0]), interpolation='bilinear')
    if yval==0:
        ax.set_title("hospital " + str(hospital))
    if hospital==0:
        ax.set_ylabel("y=" +  str(yval))

plt.savefig("camelyon17.png")


file = open('outputs', 'rb')
outputs = pickle.load(file)
file.close()


# Produce calibration curves
logits = outputs['SWAV-seed1']['OD_logits']
y = outputs['SWAV-seed1']['OD_true']

bins = np.linspace(0,1,11)
logits.softmax(1)[:,0].numpy()
bin_indices = np.around(logits.softmax(1)[:,0].numpy(), 1)

predictions = [sum(y[bin_indices == b] == 0).item() / sum(bin_indices == b).item() for b in bins]

fig, ax = plt.subplots()
ax.bar(bins, predictions, width=0.1, alpha=0.3, linewidth=1, edgecolor='black')
ax.plot(bins, bins, color='black', linestyle='--')
ax.set_title('SWAV - OD Calibration')

plt.savefig("calibration.png")
