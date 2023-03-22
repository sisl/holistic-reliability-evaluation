import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import sys, os
sys.path.append(os.path.dirname(__file__))
from load_wilds_models import *
from hre_model import ClassificationTask


def evaluate(config, model=None, results_dir="eval_results"):
    hremodel = ClassificationTask(config, model)
    
    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        logger=CSVLogger(
            results_dir, name=config["algorithm"], version=config["seed"]
        ),
        inference_mode=False,
    )

    trainer.validate(hremodel)
    trainer.test(hremodel)


def evaluate_all_seeds(algorithm, load_fn, ckpt_gen, config, Nseeds, results_dir="eval_results", args={}):
    config["algorithm"] = algorithm
    config["wilds_pretrained"] = True
    for seed in range(Nseeds):
        config["checkpoint_path"] = ckpt_gen(seed)
        config["seed"] = seed
        wilds_model = load_fn(config["checkpoint_path"], config["n_classes"], **args)
        evaluate(config, wilds_model, results_dir)