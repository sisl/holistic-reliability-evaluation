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
    
    # Decide whether or not to validate the model
    parser.add_argument("--validate", type=bool, default=True)
    
    # Decide whether or not to test the model
    parser.add_argument("--test", type=bool, default=True)
    
    return parser.parse_args()

## This function ensures that all models get evaluated on the same datasets regardless of their configs
def get_datasets(dataset):
    if dataset == "iwildcam":
        return {"test_ds_datasets": ["iwildcam-test", "iwildcam-id_test-corruption1_test"],
                "test_id_dataset": "iwildcam-id_test",
                "test_ood_datasets": ["gaussian_noise", "fmow-id_test", "rxrx1-id_test", "camelyon17-id_test"],
                "val_ds_datasets": ["iwildcam-val", "iwildcam-id_val-corruption1_val"],
                "val_id_dataset": "iwildcam-id_val",
                "val_ood_datasets": ["gaussian_noise", "fmow-id_val", "rxrx1-id_val", "camelyon17-id_val"],}
    elif dataset == "camelyon17":
        return {"test_ds_datasets": ["camelyon17-test", "camelyon17-id_test-corruption1_test"],
                "test_id_dataset": "camelyon17-id_test",
                "test_ood_datasets": ["gaussian_noise", "fmow-id_test", "rxrx1-id_test", "iwildcam-id_test"],
                "val_ds_datasets": ["camelyon17-val", "camelyon17-id_val-corruption1_val"],
                "val_id_dataset": "camelyon17-id_val",
                "val_ood_datasets": ["gaussian_noise", "fmow-id_val", "rxrx1-id_val", "iwildcam-id_val"],}
    elif dataset == "fmow":
        return {"test_ds_datasets": ["fmow-test", "fmow-id_test-corruption1_test"],
                "test_id_dataset": "fmow-id_test",
                "test_ood_datasets": ["gaussian_noise", "camelyon17-id_test", "rxrx1-id_test", "iwildcam-id_test"],
                "val_ds_datasets": ["fmow-val", "fmow-id_val-corruption1_val"],
                "val_id_dataset": "fmow-id_val",
                "val_ood_datasets": ["gaussian_noise", "camelyon17-id_val", "rxrx1-id_val", "iwildcam-id_val"],}
    elif dataset == "rxrx1":
        return {"test_ds_datasets": ["rxrx1-test", "rxrx1-id_test-corruption1_test"],
                "test_id_dataset": "rxrx1-id_test",
                "test_ood_datasets": ["gaussian_noise", "camelyon17-id_test", "fmow-id_test", "iwildcam-id_test"],
                "val_ds_datasets": ["rxrx1-val", "rxrx1-id_val-corruption1_val"],
                "val_id_dataset": "rxrx1-id_val",
                "val_ood_datasets": ["gaussian_noise", "camelyon17-id_val", "fmow-id_val", "iwildcam-id_val"],}
    else:
        raise ValueError("Dataset {} not supported".format(dataset))
    
def process_args():
    args = parse_args()
    
    # Setup the changes to the configuration
    config_args = {
        "val_dataset_length" : args.eval_size,
        "test_dataset_length" : args.eval_size,
        "calibration_method" : args.calibration_method,
        "inference_mode" : args.inference_mode,
        "data_dir" : args.data_dir,
        "num_adv" : 128,
        **get_datasets(args.dataset)
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
    return model_descriptions, config_args, args, save_dir
    

def run_evaluate():
    model_descriptions, config_args, args, save_dir = process_args()

    # Evaluate the models
    for modelfn, ver in model_descriptions:
        evaluate(modelfn(config_args), save_dir, version=ver, validate=args.validate, test=args.test)


if __name__ == "__main__":
    run_evaluate()
