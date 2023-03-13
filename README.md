# Holistic Reliability Evaluation

A toolkit for evaluating the reliability of ML models along a variety of dimensions including

* In-distribution accuracy
* Accuracy under domain shifts
* Adversarial robustness
* Uncertainty Quantification
* Out of Distribution Detection

## Installation
* install `wilds`
* install `autoattack`: (https://github.com/fra31/auto-attack)


## Running Training
Run the training code by calling
```
python training/train.py --config training/configs/camelyon-defaults.yml --seed 1234
```

Where the configuration file describes the correct parameters.

### Running Tunning sweeps
To run tuning sweeps take the following steps:
*
