import numpy as np
from .evaluation import *
import matplotlib.pyplot as plt

def report_stats(v, sigfigs=3):
    v = np.array(v)
    return f"{round(v.mean(), sigfigs)} ({round(v.std(), sigfigs)})"


def print_table(column_count, headers, table_content):
    cs = "c"*column_count
    print(f"\\begin{{tabular}}{{@{{}}{cs}@{{}}}}")
    print("\\toprule")
    for row in headers:
        print(row, "\\\\")
        print("\\midrule")

    for row in table_content:
        print(row, "\\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")

class ResultsProcessor:
    def __init__(self, models):
        self.models = models

    def group_models_by_type(self):
        model_types = dict()
        for m in self.models:
            if m.type in model_types:
                model_types[m.type].append(m)
            else:
                model_types[m.type] = [m]
        return model_types

    def datasets_with_metric(self, metric):
        datasets = dict()
        for m in self.models:
            for d in  m.results.keys():
                if metric in m.results[d]:
                    datasets[d] = None
        return list(datasets.keys())
    
    def metrics_that_contain(self, substring):
        metrics = dict()
        for m in self.models:
            for d in  m.results.keys():
                for metric in m.results[d]:
                    if substring in metric:
                        metrics[metric] = None
        return list(metrics.keys())

    def build_table(self, metrics, keys_per_type=[]):
        grouped_models = self.group_models_by_type()

        
        if len(keys_per_type)>1:
            column_count = 2
            table = []
            headers = [f"\multicolumn{{{len(keys_per_type)}}}{{c}}{{Method}}", f"\multicolumn{{{len(keys_per_type)}}}{{c}}{{}}"]
            for type in grouped_models:
                first = True
                for key in keys_per_type:
                    if first:
                        table.append(f"\multirow{{{len(keys_per_type)}}}{{*}}{{{type}}} & {key}")
                    else:
                        table.append(f" & {key}")
                    first = False
        else:
            column_count = 1
            headers = ["Method ", ""]
            table = [type for type in grouped_models]

        
        for metric in metrics:
            if len(keys_per_type):
                datasets = self.datasets_with_metric(keys_per_type[0])
            else:
                datasets = self.datasets_with_metric(metric)
            headers[0] += "& \multicolumn{{{}}}{{c}}{{{}}}".format(len(datasets), metric)
            for dataset in datasets:
                column_count += 1
                headers[1] += "& {} ".format(dataset)
                i=0
                for type in grouped_models:
                    if len(keys_per_type):
                        for key in keys_per_type:
                            t = [model.results[dataset][key][metric] for model in grouped_models[type]]
                            table[i] += " & " + report_stats(t)
                            i+=1
                    else:
                        t = [model.results[dataset][metric] for model in grouped_models[type]]
                        table[i] += " & " + report_stats(t)
                        i+=1

        print_table(column_count, headers, table)


    def robustness_table(self):
        self.build_table(['Accuracy', 'Adversarial Accuracy'])
    
    def uq_table(self):
        self.build_table(["Expected Calibration Error", *self.metrics_that_contain("Set Size (Acc")])
    
    def ood_table(self):
        #ood_metrics = ["AUROC", "AUPR-IN", "AUPR-OUT", "ACC95TPR", "FPR95TPR"]
        ood_metrics = ["AUROC", "FPR95TPR"]
        #ood_detectors = ["Max Softmax", "Energy-Based"]
        ood_detectors = ["Energy-Based"]
        self.build_table(ood_metrics, ood_detectors)

    def get_metrics(self, datasets, metric, approach=None, return_types=False):
        grouped_models = self.group_models_by_type()
        model_list = [x for v in grouped_models.values() for x in v]
        metlist=[]
        typelist=[]
        for m in model_list:
            for d in datasets:
                typelist.append(m.type)
                if approach is not None:
                    metlist.append(m.results[d][approach][metric])
                else:
                    metlist.append(m.results[d][metric])
        if return_types:
            return metlist, typelist
        else:
            return metlist
        
    def plot_error_correlation(self, dataset, filepath=None, figsize=(18.5, 10.5)):
        grouped_models = self.group_models_by_type()
        model_list = [x for v in grouped_models.values() for x in v]
        labels = [m.name() for m in model_list]

        #average blocks here
        error_correlations = [[eval_error_correlation(m1, m2, dataset) for m1 in model_list] for m2 in model_list]
        fig, ax = plt.subplots()
        fig.set_size_inches(*figsize)
        im = ax.imshow(error_correlations, vmin=0, vmax=1)
        

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        ax.set_title(f"Error Correlations for Dataset {dataset}")
        fig.tight_layout()
        if filepath is None:
            plt.show()
        else:
            plt.savefig(filepath)
                

