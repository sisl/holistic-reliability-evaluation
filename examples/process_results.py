import os
import pickle
from scipy.stats import pearsonr
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('/home/dk11/holistic-reliability-evaluation/')

from holistic_reliability_evaluation.results_processor import ResultsProcessor

results_dir = "iWild_results/"
dataset_name = "iWildCam"

def load_model(dir, filename):
    file = open(dir + filename, 'rb')
    model = pickle.load(file)
    file.close()
    return model

models = [load_model(results_dir, p) for p in os.listdir(results_dir)]

for m in models:
    if m.type == 'DEEPCORAL-COURSE':
        m.type = "deepCORAL w/ Unlabeled"
    if m.type == 'DANN-COURSE':
        m.type = "DANN"
        
elements = ['ERM', 'ERM-Augment', 'deepCORAL',  "deepCORAL w/ Unlabeled", 'IRM', 'groupDRO', 'DANN-COURSE', 'SwAV', 'PseudoLabels', 'FixMatch', 'AFN', 'Noisy-Student']
colors = ['#2f4f4f', '#7f0000', '#008000', '#00008b', '#ff8c00', '#ffff00', '#00ff00', '#00ffff', '#ff00ff', '#1e90ff', '#f5deb3', '#ff69b4']


rp = ResultsProcessor(models, elements)

for m in models:
    m.fill_hre()


def HRE_Score_metrics():    
    rp.metrics_table()


    grouped_models = rp.group_models_by_type()
    fig, ax = plt.subplots()

    xi = 0
    for type in grouped_models:
        yvals = [m.results["HRE Score"] for m in grouped_models[type]]
        xvals = [xi for m in grouped_models[type]]
        ax.scatter(xvals, yvals, color=colors[xi])
        xi = xi + 1
    plt.xticks(ticks = range(len(grouped_models)), labels=[type for type in grouped_models], rotation = 90)
    plt.ylabel("HRE Score")
    plt.title(dataset_name)
    fig.set_size_inches(6,6)
    fig.tight_layout()
    plt.savefig("HRE_Score_scatter.png")
    
def HRE_metrics_plot(m1, m2):
    grouped_models = rp.group_models_by_type()
    fig, ax = plt.subplots()

    xi = 0
    for type in grouped_models:
        xvals = [m.results[m1] for m in grouped_models[type]]
        yvals = [m.results[m2] for m in grouped_models[type]]
        ax.scatter(xvals, yvals, color=colors[xi], label=type)
        xi = xi + 1
    plt.ylabel(m2)
    plt.xlabel(m1)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(m2 + " vs " + m1 + "(" + dataset_name + ")")
    fig.set_size_inches(10,6)
    fig.tight_layout()
    plt.savefig(m2 + "_vs_" + m1 + ".png")
    
    

##making plots and charts and tables
def make_plots(dataset, datasets1, datasets2, metric1, metric2, approach1, approach2):
    plt.clf()
    ##create metrics and plots
    m1, types1 = rp.get_metrics(datasets1, metric1, return_types=True, approach=approach1)
    m2, types2 = rp.get_metrics(datasets2, metric2, return_types=True, approach=approach2)

    assert types1 == types2

    element_colors = dict(zip(elements, colors))
    types1 = np.array(types1)
    m1 = np.array(m1)
    m2 = np.array(m2)

    #actual plotting, everything above sets up parameters to grab
    for element in elements:
        plt.scatter(m1[types1==element], m2[types1==element], c=element_colors[element], label=element)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel(metric1)
    plt.ylabel(metric2)
    
    plt.gcf().set_size_inches(12,8)

    if len(datasets1) == 1:
        plt.title(metric2 + "(" + datasets2[0] + ") vs " + metric1+ "(" + datasets1[0] + ")")
        plt.tight_layout(pad=5)
        plt.savefig(metric2 + "_" + datasets2[0] + "_vs_" + metric1 + "_" + datasets1[0] + ".png")
    else:
        plt.title(metric2 + " vs " + metric1)
        plt.tight_layout(pad=5)
        plt.savefig(metric2 + "_vs_" + metric1 + ".png")
    plt.clf()

def process_results(dataset):
    plt.clf()
    models = [load_model(results_dir, p) for p in os.listdir(results_dir)]
    rp = ResultsProcessor(models)

    ## Tabular results
    rp.robustness_table()

    print("========")

    rp.uq_table()

    print("========")

    rp.ood_table()

    # Error correlations
    rp.plot_error_correlation("ID", "id_err_corr.pdf")
    rp.plot_error_correlation("Real World DS", "ds_err_corr.pdf")
    rp.plot_error_correlation("C Bar (severity 5)", "c5_err_corr.pdf")


    ## Metric correlations
    acc_datasets = ["ID", "Real World DS", "C Bar (severity 1)", "C Bar (severity 5)"]
    acc_datasets_short = ["ID", "Real Shift", "Corrupt-1", "Corrupt-5"]
    ood_datasets = ["RxRx1", "Gaussian Noise"]

    ood_approach = "Energy-Based"
    robustness_metrics = ["Accuracy", "Adversarial Accuracy"]
    uq_metrics = ["Expected Calibration Error", "Avg. Set Size (Acc. Softmax, alpha=0.2)"]
    uq_metrics_short = ["ECE", "Set Size"]
    ood_metrics = ["AUROC", "FPR95TPR"]


    names = []
    vals = []

    for m in robustness_metrics:
        for (d, dshort) in zip(acc_datasets, acc_datasets_short):
            names.append(m + " - " + dshort)
            vals.append(rp.get_metrics([d], m))

    for (m, mshort) in zip(uq_metrics, uq_metrics_short):
        for (d, dshort) in zip(acc_datasets, acc_datasets_short):
            names.append(mshort + " - " + dshort)
            vals.append(rp.get_metrics([d], m))

    for m in ood_metrics:
        for d in ood_datasets:
            names.append(m + " - " + d)
            vals.append(rp.get_metrics([d], m, approach=ood_approach))

    for (n,v) in zip(names, vals):
        print("name: ", n, " vals: ", v)

    metric_correlations = [[pearsonr(x, y)[0] for x in vals] for y in vals]
    metric_correlations
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)
    im = ax.imshow(metric_correlations, vmin=-1, vmax=1)


    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(names)), labels=names)
    ax.set_yticks(np.arange(len(names)), labels=names)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    ax.set_title(f"Metric Correlations")
    fig.tight_layout()
    #plt.show()

    plt.savefig("metric_correlations.pdf")
    plt.close()

def comparison_plots(dataset):
    ## Plot some specific relationships
    approach1=None
    approach2=None
    metric1 = "Accuracy"
    metric2 = "Accuracy"
    datasets1 = ["ID"]#, "Real World DS", "C Bar (severity 1)", "C Bar (severity 5)"]
    datasets2 = ["Real World DS"]
    make_plots(dataset, datasets1, datasets2, metric1, metric2, approach1, approach2)
    

    approach1=None
    approach2=None
    metric3 = "Accuracy"
    metric4 = "Accuracy"
    datasets3 = ["ID"]#, "Real World DS", "C Bar (severity 1)", "C Bar (severity 5)"]
    datasets4 = ["C Bar (severity 1)"]
    make_plots(dataset, datasets3, datasets4, metric3, metric4, approach1, approach2)

    approach1=None
    approach2=None
    metric1 = "Accuracy"
    metric2 = "Accuracy"
    datasets1 = ["ID"]#, "Real World DS", "C Bar (severity 1)", "C Bar (severity 5)"]
    datasets2 = ["C Bar (severity 5)"]
    make_plots(dataset, datasets1, datasets2, metric1, metric2, approach1, approach2)

    approach1=None
    approach2=None
    metric1 = "Accuracy"
    metric2 = "Expected Calibration Error"
    datasets1 = ["ID", "Real World DS", "C Bar (severity 1)", "C Bar (severity 5)"]
    datasets2 = ["ID", "Real World DS", "C Bar (severity 1)", "C Bar (severity 5)"]
    make_plots(dataset, datasets1, datasets2, metric1, metric2, approach1, approach2)

    approach1=None
    approach2=None
    metric1 = "Accuracy"
    metric2 = "Adversarial Accuracy"
    datasets1 = ["ID", "Real World DS", "C Bar (severity 1)", "C Bar (severity 5)"]
    datasets2 = ["ID", "Real World DS", "C Bar (severity 1)", "C Bar (severity 5)"]
    make_plots(dataset, datasets1, datasets2, metric1, metric2, approach1, approach2)

    metric1 = "Expected Calibration Error"
    metric2 = "AUROC"
    approach1=None
    approach2="Energy-Based"
    datasets1 = ["ID"]
    datasets2 = ["RxRx1"]
    make_plots(dataset, datasets1, datasets2, metric1, metric2, approach1, approach2)

    metric1 = "Expected Calibration Error"
    metric2 = "AUROC"
    approach1=None
    approach2="Energy-Based"
    datasets1 = ["Real World DS"]
    datasets2 = ["RxRx1"]
    make_plots(dataset, datasets1, datasets2, metric1, metric2, approach1, approach2)

    metric1 = "Expected Calibration Error"
    metric2 = "AUROC"
    approach1=None
    approach2="Energy-Based"
    datasets1 = ["C Bar (severity 5)"]
    datasets2 = ["RxRx1"]
    make_plots(dataset, datasets1, datasets2, metric1, metric2, approach1, approach2)



# process_results(dataset_name)
# comparison_plots(dataset_name)
HRE_Score_metrics()

HRE_metrics_plot("Accuracy", "Robustness")
HRE_metrics_plot("Accuracy", "UQ")
HRE_metrics_plot("Accuracy", "OOD Detection")
HRE_metrics_plot("Robustness", "UQ")
HRE_metrics_plot("Robustness", "OOD Detection")
HRE_metrics_plot("UQ", "OOD Detection")

    
