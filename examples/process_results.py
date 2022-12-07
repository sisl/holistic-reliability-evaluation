import os
import pickle

from holistic_reliability_evaluation.results_processor import ResultsProcessor

results_dir = "results/"

def load_model(dir, filename):
    file = open(dir + filename, 'rb')
    model = pickle.load(file)
    file.close()
    return model

models = [load_model(results_dir, p) for p in os.listdir(results_dir)]

models[3].results
# models[3].results["ID"]["Accuracy"]

# models[2].type

# import numpy as np
# v = np.array([0.8, 0.86])
# v.mean()

rp = ResultsProcessor(models)

rp.robustness_table()

print("========")

rp.uq_table()

print("========")

rp.ood_table()