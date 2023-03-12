# Holistic Reliability Evaluation

A toolkit for evaluating the reliability of ML models along a variety of dimensions including

* In-distribution accuracy
* Accuracy under domain shifts
* Adversarial robustness
* Uncertainty Quantification
* Out of Distribution Detection

## Installation
* install `wilds`
* install `augmentation-corruption` and its dependencies (https://github.com/facebookresearch/augmentation-corruption/tree/fbr_main/imagenet_c_bar/utils).
    * Add the folders `augmentation-corruption/imagenet_c_bar` and `augmentation-corruption/imagenet_c_bar/utils` to `PYTHONPATH`


## Running Training
Run the training code by calling
```
python training/train.py --config training/configs/camelyon-defaults.yml --seed 1234
```

Where the configuration file describes the correct parameters.

### Running Tunning sweeps
To run tuning sweeps take the following steps:
*
