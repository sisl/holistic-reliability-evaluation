import numpy as np

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

        
        if len(keys_per_type):
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
        self.build_table(["Expected Calibration Error", *self.metrics_that_contain("Set Size")])
    
    def ood_table(self):
        ood_metrics = ["AUROC", "AUPR-IN", "AUPR-OUT", "ACC95TPR", "FPR95TPR"]
        ood_detectors = ["Max Softmax", "Energy-Based"]
        self.build_table(ood_metrics, ood_detectors)

