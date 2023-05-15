import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import argparse

import sys, os
sys.path.append(os.path.dirname(__file__))
from utils import *
from pretrained_models import *

def evaluate(
    model,
    results_dir="eval_results",
    validate=True,  # Whether or not to call trainer.validate()
    test=True,  # Whether or not to call trainer.test()
    name=None,  # override config["algorithm"] for naming
    version=None,  # override config["seed"] for versioning
    return_results=False,  # Whether or not to return the dataframe containing results
    skip_if_exists=True,  # Whether or not to skip if results already exist
):
    # Set the name of the run (for where to save it)
    if name is None:
        name = model.config["algorithm"]

    # Set the version of this model (typically either the seed for wilds pretrained or the wandb hash)
    if version is None:
        version = model.config["seed"]
    
    logger = CSVLogger(results_dir, name=name, version=version)
    
    # Check if logger.log_dir already exists
    print("=>  Going to write: ", logger.log_dir)
    if os.path.exists(logger.log_dir) and skip_if_exists:
        print(f"Results for {name}/{version} already exist, skipping...")
        return

    trainer = pl.Trainer(
        accelerator=model.config["accelerator"],
        devices=model.config["devices"],
        logger=logger,
        inference_mode=model.config["inference_mode"],
    )

    if validate:
        trainer.validate(model)

    if test:
        trainer.test(model)

    if return_results:
        return load_results(trainer.logger.log_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    # Name of the wilds datset to evaluate
    parser.add_argument("--dataset")
    
    # Set if we are using the wilds pretrained models, if self-trained use false
    parser.add_argument("--wilds_pretrained", type=bool, default=False)
    
    # Number of seeds for that evaluation. If not supplied the default is used
    parser.add_argument("--Nseeds", type=int, default=-1)
    
    # Directory where the models are stored
    parser.add_argument("--model_dir")
    
    # Directory where the results should be stored
    parser.add_argument("--save_dir")
    
    # Whether or not to use inference mode
    parser.add_argument("--inference_mode", type=bool, default=False)
    
    # Number of samples to evaluate on
    parser.add_argument("--eval_size", type=int, default=1024)
    
    # Set the calibration method to use
    parser.add_argument("--calibration_method", default="none")
    
    # Set the data directory (it might have been set by someone else in the config we load)
    parser.add_argument("--data_dir", default="/scratch/users/acorso/data/")
    
    return parser.parse_args()


def run_evaluate():
    args = parse_args()
    
    # Setup the changes to the configuration
    config_args = {
        "val_dataset_length" : args.eval_size,
        "test_dataset_length" : args.eval_size,
        "calibration_method" : args.calibration_method,
        "inference_mode" : args.inference_mode,
        "data_dir" : args.data_dir
    }
    
    if args.inference_mode:
        config_args["num_adv"] = 0
        config_args["w_sec"] = 0.0
        
    # Get the dataset-specific model directory
    dataset_dir = os.path.join(args.model_dir, args.dataset)

    ## Load the model descriptions
    if args.wilds_pretrained:
        print(f"Loading WILDS models for dataset {args.dataset}")
        if args.dataset == "camelyon17":
            model_descriptions = camelyon17_pretrained_models(dataset_dir, args.Nseeds)
        elif args.dataset == "iwildcam":
            model_descriptions = iwildcam_pretrained_models(dataset_dir, args.Nseeds)
        elif args.dataset == "fmow":
            model_descriptions = fmow_pretrained_models(dataset_dir, args.Nseeds)
        elif args.dataset == "rxrx1":
            model_descriptions = rxrx1_pretrained_models(dataset_dir, args.Nseeds)
        else:
            raise ValueError("Dataset {} not supported".format(args.dataset))
    else:
        print(f"Loading self-trained models for dataset {args.dataset}")
        model_descriptions = load_model_descriptions(args.model_dir, args.dataset)

    # combine results_dir with dataset name
    save_dir = os.path.join(args.save_dir, args.dataset)

    # Evaluate the models
    for modelfn, ver in model_descriptions:
        evaluate(modelfn(config_args), save_dir, version=ver)


if __name__ == "__main__":
    run_evaluate()
