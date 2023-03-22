# Holistic Reliability Evaluation

A toolkit for evaluating the reliability of ML models along a variety of dimensions including

* In-distribution performance
* Performance under domain shifts
* Adversarial robustness
* Uncertainty Quantification
* Out of Distribution Detection

## Installation
* install `wilds`
* install `autoattack`: (https://github.com/fra31/auto-attack)

## Evaluating pre-trained Wilds models
* Download the pretrained wilds models
* Run the script to evaluate them and store the results. For each DATASET run
```
python holistic_reliability_evaliation/evaluate_wilds_DATASET.py
```
## Running Training
Run the training code by calling
```
python holistic_reliability_evaliation/train.py --config configs/camelyon-defaults.yml --seed 1234
```

Where the configuration file describes the correct parameters.

### Running Tunning sweeps
To run tuning sweeps take the following steps:
*
