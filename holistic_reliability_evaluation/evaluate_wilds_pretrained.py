import argparse
import sys, os

sys.path.append(os.path.dirname(__file__))
from wilds_models import *
from utils import *
from evaluate import *


def evaluate_all_seeds(
    model_desc, results_dir="eval_results", inference_mode=False, eval_size=1024
):
    config = model_desc["config"]
    filename_fn = model_desc["filename_fn"]
    load_fn = model_desc["load_fn"]
    args = model_desc["args"]
    
    config["val_dataset_length"] = eval_size
    config["test_dataset_length"] = eval_size

    # Check the inference mode, if True, disable adversarial evaluation
    if inference_mode:
        config["num_adv"] = 0
        config["w_sec"] = 0.0

    # Set the algorithm name and an indicator that this was one of the wilds pretrained models
    config["algorithm"] = model_desc["name"]
    config["wilds_pretrained"] = True

    # Loop through each seed and evaluate
    for seed in range(model_desc["Nseeds"]):
        config["checkpoint_path"] = filename_fn(seed)
        config["seed"] = seed
        wilds_model = load_fn(config["checkpoint_path"], config["n_classes"], **args)
        evaluate(config, wilds_model, results_dir)


## Setup the arguments that can be handled
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")  # Name of the wilds datset to evaluate
parser.add_argument(
    "--Nseeds", type=int, default=-1
)  # Number of seeds for that evaluation. If not supplied the default is used
parser.add_argument("--model_dir") # Directory where the models are stored
parser.add_argument("--save_dir") # Directory where the results should be stored
parser.add_argument("--inference_mode", type=bool, default=False) # Whether or not to use inference mode
parser.add_argument("--eval_size", type=int, default=1024) # Number of samples to evaluate on 
args = parser.parse_args()

# Get the dataset-specific model directory
model_dir = os.path.join(args.model_dir, args.dataset)

## Load the model descriptions
print("Loading models for dataset {}".format(args.dataset))
if args.dataset == "camelyon17":
    model_descriptions = camelyon17_pretrained_models(model_dir, args.Nseeds)
elif args.dataset == "iwildcam":
    model_descriptions = iwildcam_pretrained_models(model_dir, args.Nseeds)
elif args.dataset == "fmow":
    model_descriptions = fmow_pretrained_models(model_dir, args.Nseeds)
elif args.dataset == "rxrx1":
    model_descriptions = rxrx1_pretrained_models(model_dir, args.Nseeds)
else:
    raise ValueError("Dataset {} not supported".format(args.dataset))

# combine results_dir with dataset name
save_dir = os.path.join(args.save_dir, args.dataset)

# Evaluate the models
for model_desc in model_descriptions:
    print("Evaluating models of type {}".format(model_desc["name"]))
    evaluate_all_seeds(model_desc, save_dir, args.inference_mode, args.eval_size)
