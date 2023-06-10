import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import random
from torch.utils.data import DataLoader
from hre_datasets import get_subset

import numpy as np

import argparse
import sys, os

sys.path.append(os.path.dirname(__file__))
from pretrained_models import *
from utils import *
from evaluate import *

## Class to ensemble the models
class EnsembleClassifier(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        num_models = len(models)
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        self.NVal = 1024
        self.device = torch.device("cuda")
        self.fit_seed = 1234
        self.val_seed = 5678

    def forward(self, x):
        outputs = []

        for i, model in enumerate(self.models):
            output = model(x)
            weight = self.weights[i]
            output = output * weight
            outputs.append(output)

        ensemble_output = torch.stack(outputs).sum(dim=0)
        
        return ensemble_output

    def val_dataset(self, seed):
        random.seed(seed)
        d = DataLoader(get_subset(self.models[0].val_id, self.NVal), batch_size=32, num_workers=32)
        random.seed()
        return d
    
    def fit_weights(self):
        validation_data = self.val_dataset(self.fit_seed)
        device = self.device
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.weights], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')
        
        logits_lists = [[] for _ in range(len(self.models))]
        labels_list = []
        
        progress_bar = tqdm(validation_data)  # Create a tqdm progress bar
        for batch in progress_bar:
            x, y = batch[0].to(device), batch[1].to(device)
            labels_list.append(y)
            
            for (model, logits_list) in zip(self.models, logits_lists):
                model.eval()
                with torch.no_grad():
                    logits_list.append(model(x))
                    
        
        # Create tensors
        logits_list = [torch.cat(logits_list).to(device) for logits_list in logits_lists]
        labels_list = torch.cat(labels_list).to(device)
        
        def _eval():
            optimizer.zero_grad()  # Reset gradients
            
            # compute logits with the weights
            weighted_logits_list = []

            for i, logits in enumerate(logits_list):
                weighted_logits = logits * self.weights[i]
                weighted_logits_list.append(weighted_logits)

            logits = torch.stack(weighted_logits_list).sum(dim=0)
            loss = criterion(logits, labels_list)
            loss.backward()
            return loss.item()
        optimizer.step(_eval)
        
    def eval_accuracy(self):
        self.eval()
        val_dataset = self.val_dataset(self.val_seed)
        correct = 0
        total = 0
        progress_bar = tqdm(val_dataset)  # Create a tqdm progress bar
        
        for batch in progress_bar:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            with torch.no_grad():
                logits = self.forward(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
                
            # Update the progress bar with the current iteration count
            progress_bar.set_description(f"Accuracy: {correct / total:.4f}")
        
        return correct / total
        

def select_best_ensemble(models, config_args, Nensemble, Ntrials):
    random.seed()
    best_accuracy = 0
    best_ensemble = None
    for i in range(Ntrials):
        indices = random.sample(range(len(models)), Nensemble)
        print(f"Trial {i}: {indices}")
        model_subset = [models[i][0](config_args).to(torch.device("cuda")) for i in indices]
        ens = EnsembleClassifier(model_subset)
        ens.fit_weights()
        acc = ens.eval_accuracy()
        if acc > best_accuracy:
            print(f"New best accuracy: {acc}")
            best_accuracy = acc
            best_ensemble = ens
            
    return best_ensemble
    

def eval_best_ensemble(models, config_args, Nensemble, Ntrials, save_dir, validate, test):
    ens = select_best_ensemble(models, config_args, Nensemble, Ntrials)
    config = copy.deepcopy(ens.models[0].config)
    config["algorithm"] = "Ensemble_" + str(Nensemble)
    version = "_".join([m.config["algorithm"] + "_" + str(m.config["seed"]) for m in ens.models])

    print(f"Best ensemble: {version}")
    m = ClassificationTask(config, ens)
    evaluate(m, save_dir, validate=validate, test=test, version=version)


def run_ensemble():
    model_descriptions, config_args, args, save_dir = process_args()
    config_args["batch_size"] = 32 # Batch size (we are taking gradients with all models, so this sometimes need to be reduced)
    min_ensembles = 1 # Minimum number of models in the ensemble
    max_ensembles = 5 # Maximum number of models in the ensemble
    Ntrials = 50 # Number of ensembles to contruct and measure id-val performance
    Nrepeats = 3 # Number of times to repeat the full experiment
    for i in range(Nrepeats):
        print(f"=======> Repeat {i}")
        for Nensemble in range(min_ensembles, max_ensembles + 1):
            print(f"=======> Ensemble size {Nensemble}")
            eval_best_ensemble(model_descriptions, config_args, Nensemble, Ntrials, save_dir, args.validate, args.test)        
    
if __name__ == "__main__":
    run_ensemble()

