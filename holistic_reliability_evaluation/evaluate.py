import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import sys, os
sys.path.append(os.path.dirname(__file__))
from hre_model import ClassificationTask
from utils import *


def evaluate(
    config,
    model=None,
    results_dir="eval_results",
    inference_mode=False, 
    validate=True, # Whether or not to call trainer.validate()
    test=True, # Whether or not to call trainer.test()
    name=None, # override config["algorithm"] for naming
    version=None, # override config["seed"] for versioning
    return_results=False, # Whether or not to return the dataframe containing results
):
    hremodel = ClassificationTask(config, model)

    if name is None:
        name = config["algorithm"]

    if version is None:
        version = config["seed"]

    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        logger=CSVLogger(results_dir, name=name, version=version),
        inference_mode=inference_mode,
    )

    if validate:
        trainer.validate(hremodel)

    if test:
        trainer.test(hremodel)
    
    if return_results:
        return load_results(trainer.logger.log_dir)
    