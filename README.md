# Holistic Reliability Evaluation

A toolkit for evaluating the reliability of ML models along a variety of dimensions including

* In-distribution performance
* Performance under domain shifts
* Adversarial robustness
* Uncertainty Quantification
* Out of Distribution Detection

## Installation
* install `wilds`
* install `autoattack` (https://github.com/fra31/auto-attack)
* intall the packages in `requirements.txt`

## Evaluating pre-trained Wilds models
* Download the pretrained wilds models and save them in a directory where each dataset has its own folder. E.g. `/scratch/users/acorso/wilds_models/` has folders `camelyon17`, `iwildcam`, etc.
* For a quick check that things are running well use
```
python evaluate_wilds_pretrained.py --dataset=camelyon17 --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/ --inference_mode=True --eval_size=12 --Nseeds=1
```
* To evaluate all of the models (with the recommeneded 1024 validation samples) remove the last three arguments and run, e.g.
```
python evaluate_wilds_pretrained.py --dataset=camelyon17 --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/
```

## Evaluting ensembles of pre-trained Wilds models
* Assuming the wilds models have been downloaded, make a call like
```
python ensembles.py --dataset=camelyon17 --ensemble_builder=cumulative --algorithm=erm --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/
```
* Different ensemble builders include `cumulative`, which incrementally adds models from a single aglorithm (`erm` in the above example), `greedy`, where models are added one by one, maximizing the validation hre score, and `random`, where random model subsets are chosen and evaluated.
* For the `cumulative` ensemble builder, specify the `algorithm` property
* For the `greedy` ensemble builder, specify the `n_greedy_iterations`, which are the number of outside iterations of the greedy algorithm and `n_samples`, which is the number of new models to try to add to the best ensemble
* For the `random` ensemble builder specify the `n_samples`, which is the number of random subsets are tried. 

To run all of the ensemble experiments you can use the following commands:
```
python ensembles.py --dataset=camelyon17 --ensemble_builder=cumulative --algorithm=erm --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/
python ensembles.py --dataset=camelyon17 --ensemble_builder=greedy --n_greedy_iterations=10 --n_samples=20 --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/
python ensembles.py --dataset=camelyon17 --ensemble_builder=random --n_samples=50 --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/

python ensembles.py --dataset=iwildcam --ensemble_builder=cumulative --algorithm=erm --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/
python ensembles.py --dataset=iwildcam --ensemble_builder=greedy --n_greedy_iterations=10 --n_samples=20 --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/
python ensembles.py --dataset=iwildcam --ensemble_builder=random --n_samples=50 --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/

python ensembles.py --dataset=fmow --ensemble_builder=cumulative --algorithm=erm --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/
python ensembles.py --dataset=fmow --ensemble_builder=greedy --n_greedy_iterations=10 --n_samples=20 --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/
python ensembles.py --dataset=fmow --ensemble_builder=random --n_samples=50 --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/

python ensembles.py --dataset=rxrx1 --ensemble_builder=cumulative --algorithm=erm --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/
python ensembles.py --dataset=rxrx1 --ensemble_builder=greedy --n_greedy_iterations=10 --n_samples=20 --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/
python ensembles.py --dataset=rxrx1 --ensemble_builder=random --n_samples=50 --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/
```

## Running Training
Run the training code by calling
```
python holistic_reliability_evaluation/train.py --config configs/camelyon-defaults.yml --seed 1234
```

Where the configuration file describes the correct parameters.

### Running Tunning sweeps
To run tuning sweeps take the following steps:
*
