# Holistic Reliability Evaluation
Code and results for *A Holistic Assessment of the Reliability of Machine Learning
Systems*

This code provides a unified toolkit for evaluating the reliability of ML models along a variety of dimensions including
* In-distribution performance
* Performance under distribution shifts
* Adversarial robustness (using [RobustBench](https://robustbench.github.io/))
* Uncertainty Quantification
* Out of Distribution Detection (using [pytorch-ood](https://pytorch-ood.readthedocs.io/en/latest/index.html))

## Installation Instructions
Make sure you have poetry installed by following the instructions [here](https://python-poetry.org/docs/). Then,
1. clone this repository and submodules (!)
```
git clone --recurse-submodules git@github.com:ancorso/holistic-reliability-evaluation.git
```
2. install virtual environment using poetry
```
cd holistic-reliability-evaluation
poetry install
```
3. activate environment and test installation
```
poetry shell
python -c "from holistic_reliability_evaluation.hre_model import HREModel"
```

## File Descriptions
**Configs/Runners**
* `evaluate_all_models.sh` - Contains commands to evaluate all pretrained models both before and after temperature scaling. It assumes that the WILDS pre-trained models and the ones we trained are in different folders (with different folder structures)
* `evaluate_ensebles.sh` - Contains the commands to construct and evaluate ensemebles for each of the datasets. Some parameters for these experiments are in `holistic_reliability_evaluation/evaluate_ensembles.py`
* `fine_tune_experiments.py` - The contains the python code that ran the fine-tuning experiments. It relies on a default config file and then directly updates the hyperparameters. Designed to be easy to restart in case training crashes. Swap out the dataset name for each experiment. 

**Core Sourcecode (`holistic_reliability_evaluation` folder)**
* `hre_model.py` - Contains the pytorch-lightning model description with all of the evalution techniques implemented for classification. 
* `hre_dataset.py` - Constructs a pytorch `Dataset` out of a set of datasets representing in-distribtion data, distribution-shifted data, and out-of-distribution data.
* `pretrained_models.py` - Code to load in pre-trained models, both from the WILDS benchmark and the ones we trained
* `train.py` - Train a model from a config file
* `evaluate.py` - Run an evaluation on a pre-trained model

**Results and Analysis**
* `results` - has all of the evaluation results of the models we evaluated. Folders with `_calibrated` used temperature scaling. `self_trained` models were trained by us while `wilds_` came from the WILDS benchmark
*  `analysis/alg_comparisons.jl` - Produces all of the figures for the paper using the `results`
* `analysis/paper_figures` - Contains a pdf copy of all of the figures in the paper


## Evaluating pre-trained models
To evaluate a set of pretrained models, use the `holistic_reliability_evaluation/evaluate.py` file with commands such as
```
python holistic_reliability_evaluation/evaluate.py --wilds_pretrained=true --dataset=iwildcam --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/wilds_pretrained
```
or, for a non-WILDS-pretrained example:
```
python holistic_reliability_evaluation/evaluate.py --dataset=iwildcam --model_dir=/mnt/data/acorso/results --save_dir=results/self_trained
```

The full set of options are as follows:
```
# Name of the wilds datset to evaluate (e.g. "camelyon17", "iwildcam", "fmow")
parser.add_argument("--dataset")

# Set if we are using the wilds pretrained models, if self-trained use false
parser.add_argument("--wilds_pretrained", type=bool, default=False)

# Number of seeds for that evaluation. If not supplied the default is used
parser.add_argument("--Nseeds", type=int, default=-1)

# Directory where the models are stored
parser.add_argument("--model_dir")

# Directory where the results should be stored
parser.add_argument("--save_dir")

# Whether or not to use inference mode (which is faster but disables adversarial robustness and ODIN evaluations)
parser.add_argument("--inference_mode", type=bool, default=False)

# Number of samples to evaluate on
parser.add_argument("--eval_size", type=int, default=1024)

# Set the calibration method to use, options include "none" and "temperature_scaling"
parser.add_argument("--calibration_method", default="none")

# Set the data directory (it might have been set by someone else in the config we load)
parser.add_argument("--data_dir", default="/scratch/users/acorso/data/")

# Decide whether or not to validate the model (i.e. use the validation set)
parser.add_argument("--validate", type=bool, default=True)

# Decide whether or not to test the model (i.e. use the test set)
parser.add_argument("--test", type=bool, default=True)
```

## Evaluting ensembles of pre-trained Wilds models
To run the ensembling experiments, use the `holistic_reliability_evaluation/evalute_ensembles.py` file, for example
```
python holistic_reliability_evaluation/evaluate_ensembles.py --validate=False --wilds_pretrained=true --dataset=iwildcam --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/wilds_pretrained
```

All of the same parameters used to evaluate individual models still apply. The parameters that influence the experiments are currently hardcoded (lines 143-147, shown below):
```
config_args["batch_size"] = 32 # Batch size (we are taking gradients with all models, so this sometimes need to be reduced)
min_ensembles = 1 # Minimum number of models in the ensemble
max_ensembles = 5 # Maximum number of models in the ensemble
Ntrials = 50 # Number of ensembles to contruct and measure id-val performance
Nrepeats = 3 # Number of times to repeat the full experiment
```

## Running Training
Ensure you have the wilds datasets downloaded (use `download_wilds_data.py` to do so)

Run the training code by calling
```
python holistic_reliability_evaluation/train.py --config configs/camelyon-defaults.yml --seed 1234
```

Where the configuration file describes the correct parameters.

## Running Tuning Sweeps with WandB
To run tuning sweeps with wandb take the following steps:
* Create your base config file for the dataset and techniques of interest, for example, specifying the data aug technique in train_transforms parameter (e.g. `camelyon17-augmix.yml`)
* Create your config sweep file, which references the base config file you made in the previous step and specifies other parameters you wish to conduct your training sweep over (e.g. `camelyon17-augmix-sweep.yml`)
* Navigate to the appropriate results folder for the tuning results to be stored in e.g. `/scratch/users/acorso/results`
* Intialize the sweep with `wandb sweep --project camelyon17-train ~/holistic-reliability-evaluation/configs/camelyon17-augmix-sweep.yml`
* Make note of the sweep id (e.g. `cquza740`)
* Run sweep agent with e.g. `wandb agent corso-stanford/camelyon17-train/cquza740`
