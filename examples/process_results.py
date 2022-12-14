import os
import pickle
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

from holistic_reliability_evaluation.results_processor import ResultsProcessor

results_dir = "results/"

def load_model(dir, filename):
    file = open(dir + filename, 'rb')
    model = pickle.load(file)
    file.close()
    return model

models = [load_model(results_dir, p) for p in os.listdir(results_dir)]

rp = ResultsProcessor(models)

## Tabular results
# rp.robustness_table()

# print("========")

# rp.uq_table()

# print("========")

# rp.ood_table()

## Error correlations
# rp.plot_error_correlation("ID", "id_err_corr.pdf")
# rp.plot_error_correlation("Real World DS", "ds_err_corr.pdf")
# rp.plot_error_correlation("C Bar (severity 5)", "c5_err_corr.pdf")


# ##
# acc_datasets = ["ID", "Real World DS", "C Bar (severity 1)", "C Bar (severity 5)"]
# acc_datasets_short = ["ID", "Real Shift", "Corrupt-1", "Corrupt-5"]
# ood_datasets = ["RxRx1", "Gaussian Noise"]

# ood_approach = "Energy-Based"
# robustness_metrics = ["Accuracy", "Adversarial Accuracy"]
# uq_metrics = ["Expected Calibration Error", "Avg. Set Size (Acc. Softmax, alpha=0.1)"]
# uq_metrics_short = ["ECE", "Set Size"]
# ood_metrics = ["AUROC", "FPR95TPR"]


# names = []
# vals = []

# for m in robustness_metrics:
#     for (d, dshort) in zip(acc_datasets, acc_datasets_short):
#         names.append(m + " - " + dshort)
#         vals.append(rp.get_metrics([d], m))

# for (m, mshort) in zip(uq_metrics, uq_metrics_short):
#     for (d, dshort) in zip(acc_datasets, acc_datasets_short):
#         names.append(mshort + " - " + dshort)
#         vals.append(rp.get_metrics([d], m))

# for m in ood_metrics:
#     for d in ood_datasets:
#         names.append(m + " - " + d)
#         vals.append(rp.get_metrics([d], m, approach=ood_approach))

# for (n,v) in zip(names, vals):
#     print("name: ", n, " vals: ", v)

# metric_correlations = [[pearsonr(x, y)[0] for x in vals] for y in vals]
# metric_correlations
# fig, ax = plt.subplots()
# fig.set_size_inches(18.5, 10.5)
# im = ax.imshow(metric_correlations, vmin=-1, vmax=1)


# # Show all ticks and label them with the respective list entries
# ax.set_xticks(np.arange(len(names)), labels=names)
# ax.set_yticks(np.arange(len(names)), labels=names)

# cbar = ax.figure.colorbar(im, ax=ax)
# cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

# ax.set_title(f"Metric Correlations")
# fig.tight_layout()
# # plt.show()

# plt.savefig("metric_correlations.pdf")


## Plot some specific relationships

# approach1=None
# approach2=None
# metric1 = "Accuracy"
# metric2 = "Expected Calibration Error"

# metric1 = "Accuracy"
# metric2 = "Adversarial Accuracy"

# datasets1 = ["ID", "Real World DS", "C Bar (severity 1)", "C Bar (severity 5)"]
# datasets2 = ["ID", "Real World DS", "C Bar (severity 1)", "C Bar (severity 5)"]

# metric1 = "Accuracy"
# metric2 = "Accuracy"

# datasets1 = ["ID"]#, "Real World DS", "C Bar (severity 1)", "C Bar (severity 5)"]
# # datasets2 = ["Real World DS"]
# # datasets2 = ["C Bar (severity 1)"]
# datasets2 = ["C Bar (severity 5)"]


metric1 = "Expected Calibration Error"
metric2 = "AUROC"
approach1=None
approach2="Energy-Based"

# datasets1 = ["ID"]
datasets1 = ["C Bar (severity 5)"]
datasets2 = ["RxRx1"]


m1, types1 = rp.get_metrics(datasets1, metric1, return_types=True, approach=approach1)
m2, types2 = rp.get_metrics(datasets2, metric2, return_types=True, approach=approach2)

assert types1 == types2

elements = ['deepCORAL', 'groupDRO', 'ERM-Augment', 'SwAV', 'ERM', 'PseudoLabels', 'IRM', 'FixMatch']

# The colors to use for each element
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#ffbb78", "#f781bf", "#aec7e8"]

element_colors = dict(zip(elements, colors))
types1 = np.array(types1)
m1 = np.array(m1)
m2 = np.array(m2)


for element in elements:
    plt.scatter(m1[types1==element], m2[types1==element], c=element_colors[element], label=element)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout(pad=5)
plt.xlabel(metric1)
plt.ylabel(metric2)

if len(datasets1) == 1:
    plt.title(metric2 + "(" + datasets2[0] + ") vs " + metric1+ "(" + datasets1[0] + ")")
    plt.savefig(metric2 + "_" + datasets2[0] + "_vs_" + metric1 + "_" + datasets1[0] + ".png")
else:
    plt.title(metric2 + " vs " + metric1)
    plt.savefig(metric2 + "_vs_" + metric1 + ".png")

