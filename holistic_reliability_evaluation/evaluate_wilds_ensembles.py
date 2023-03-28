import torch
import torch.nn as nn
import numpy as np

import argparse
import sys, os

sys.path.append(os.path.dirname(__file__))
from wilds_models import *
from utils import *
from evaluate import *

## Class to ensemble the models
class EnsembleClassifier(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        outputs = []

        for model in self.models:
            output = model(x)
            # probabilities = self.softmax(output)
            outputs.append(output)

        ensemble_output = torch.stack(outputs).mean(dim=0)
        
        return ensemble_output

# Function to evaluate the hre score of an ensemble of models
def eval_hre(model_set, save_dir, eval_size=1024):
    # Pull the first config file
    config = model_set[0][0]["config"]
    
    models = []
    names = []
    for (model_desc, seed) in model_set:
        load_fn = model_desc["load_fn"]
        filename_fn = model_desc["filename_fn"]
        args = model_desc["args"]
        model = load_fn(filename_fn(seed), config["n_classes"], **args)
        models.append(model)
        names.append(model_desc["name"] + "_" + str(seed))
    
    # Ensemble the models
    model = EnsembleClassifier(models)
    
    # join the names
    model_name = "_".join(names)
    
    # Set the seed to 0, since we are combining models
    config["seed"] = 0 
    
    # Disable adversarial evaluatiuon for now
    config["num_adv"] = 0
    config["w_sec"] = 0.0
    
    config["algorithm"] = model_name
    
    # Set the evaluation size
    config["val_dataset_length"] = eval_size
    config["val_batch_size"] = eval_size
    config["test_dataset_length"] = eval_size
    
    # Evaluate and return the result of the val_hre_score
    res_dict = evaluate(config, model, save_dir, inference_mode=True, return_results=True)
    return float(res_dict[0]["val_hre_score"])

def cumulative_ensemble(model_descriptions, name, save_dir, eval_size=1024):
    model_set = []
    best_model_set = []
    best_hre = 0
    
    # Get the model description for the given name
    md = next(md for md in model_descriptions if md["name"] == name)
    
    for i in range(md["Nseeds"]):
        model_set.append((md, i))
        
        # Evaluate the new model set 
        hre = eval_hre(model_set, save_dir, eval_size)
        # If there is an improvement, update the best model set 
        if hre > best_hre:
            best_hre = hre
            best_model_set = model_set.copy()
            print(f"New best hre: {best_hre}")
            print(f"New best model set size: {len(best_model_set)}")
    return best_model_set, best_hre
    
# Tries adding one model at a time, and keeps the model that leads to the best val hre score
def greedy_ensemble(model_descriptions, n_greedy_iterations, n_samples, save_dir, eval_size=1024):
    best_model_set = []
    best_hre = 0

    # Number of iterations of greedily adding models
    for k in range(n_greedy_iterations):
        last_model_set = best_model_set.copy()
        found_better = False
        for i in range(n_samples):
            # Start fresh from the best model set of the last iteration
            trial_model_set = last_model_set.copy()
            
            # Get a random new model (by specifying the model index and seed)
            model_index = np.random.randint(len(model_descriptions))
            seed = np.random.randint(model_descriptions[model_index]["Nseeds"])
            
            # Add the new model to the model_set
            trial_model_set.append((model_descriptions[model_index], seed))
            
            # Evaluate the new model set 
            hre = eval_hre(trial_model_set, save_dir, eval_size)
            
            # If there is an improvement, update the best model set 
            if hre > best_hre:
                best_hre = hre
                best_model_set = trial_model_set.copy()
                found_better = True
                print(f"New best hre: {best_hre}")
                print(f"New best model set size: {len(best_model_set)}")
        if not found_better:
            break
                
    return best_model_set, best_hre

# Tries random subsets of models and picks the subset with the highest val hre score
def random_ensemble(model_descriptions, n_samples, save_dir, eval_size=1024, max_subset_size=10):
    ## Randomly select subsets (up to 10) and compute hre score
    best_model_set = {}
    best_hre = 0
    for k in range(n_samples):
        trial_model_set = []
        
        # Sample number of models in the subset
        Nmodels = np.random.randint(max_subset_size)+1
        for i in range(Nmodels):
            # Get a random new model (by specifying the model index and seed)
            model_index = np.random.randint(len(model_descriptions))
            seed = np.random.randint(model_descriptions[model_index]["Nseeds"])
            
            # Add the new model to the model_set
            trial_model_set.append((model_descriptions[model_index], seed))
        
        
        # Evaluate the new model set 
        hre = eval_hre(trial_model_set, save_dir, eval_size)
            
        # If there is an improvement, update the best model set 
        if hre > best_hre:
            best_hre = hre
            best_model_set = trial_model_set.copy()
            print(f"New best hre: {best_hre}")
            print(f"New best model set size: {len(best_model_set)}")
                
    return best_model_set, best_hre
    

## Setup the arguments that can be handled
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")  # Name of the wilds datset to evaluate
parser.add_argument("--model_dir") # Directory where the models are stored
parser.add_argument("--save_dir") # Directory where the results should be stored
parser.add_argument("--eval_size", type=int, default=1024) # Number of samples to evaluate on 
parser.add_argument("--ensemble_builder") # cumulative, greedy, random
parser.add_argument("--algorithm", default="not_specificed") # Algorithm used for the cumulative ensemble
parser.add_argument("--n_greedy_iterations", type=int, default=10)  # Number of greedy iterations
parser.add_argument("--n_samples", type=int, default=10) # Number of inner iterations (samples) for the greedy ensemble, or total samples for random 
args = parser.parse_args()

# Get the dataset-specific model directory
model_dir = os.path.join(args.model_dir, args.dataset)

## Load the model descriptions
if args.dataset == "camelyon17":
    model_descriptions = camelyon17_pretrained_models(model_dir)
elif args.dataset == "iwildcam":
    model_descriptions = iwildcam_pretrained_models(model_dir)
elif args.dataset == "fmow":
    model_descriptions = fmow_pretrained_models(model_dir)
elif args.dataset == "rxrx1":
    model_descriptions = rxrx1_pretrained_models(model_dir)
else:
    raise ValueError("Dataset {} not supported".format(args.dataset))

# combine results_dir with dataset name
save_dir = os.path.join(args.save_dir, args.ensemble_builder, args.dataset)

# build and evalute the ensemebles using the specified approach
if args.ensemble_builder == "cumulative":
    best_set, best_score = cumulative_ensemble(model_descriptions, args.algorithm, save_dir, args.eval_size)
elif args.ensemble_builder == "greedy":
    best_set, best_score = greedy_ensemble(model_descriptions, args.n_greedy_iterations, args.n_samples, save_dir, args.eval_size)
elif args.ensemble_builder == "random":
    best_set, best_score = random_ensemble(model_descriptions, args.n_samples, save_dir, args.eval_size)

print("Best model set:", best_set, " hre: ", best_score)
