#!/usr/bin/env sh

DEFAULT_CONFIGS=('iwildcam-defaults.yml' 'fmow-defaults.yml' 'camelyon17-defaults.yml' 'rxrx1-defaults.yml')
TMPD="tmp"

for iter in {1..1}; do
    for DEFAULT_CONFIG in "${DEFAULT_CONFIGS[@]}"; do
        ./configs/bin/yq eval '.save_folder = "/scratch/users/romeov/holistic-reliability-evaluation/results",
                               .data_dir = "/scratch/users/romeov/holistic-reliability-evaluation/data",
                               .adversarial_training_method = "PGD",
                               .adversarial_training_eps = "3/255",
                               .algorithm = "adversarial_sweep",
                               .max_num_workers = 16' configs/$DEFAULT_CONFIG > $TMPD/$DEFAULT_CONFIG
        sbatch submit_config.sh $TMPD/$DEFAULT_CONFIG

        ./configs/bin/yq e '.adversarial_training_eps = "1/255"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_1.$iter.yml
        sbatch submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_1.$iter.yml $iter
        ./configs/bin/yq e '.adversarial_training_eps = "8/255"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_2.$iter.yml
        sbatch submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_2.$iter.yml $iter


        ./configs/bin/yq e '.adversarial_training_method = "FGSM"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_3.$iter.yml
        sbatch submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_3.$iter.yml $iter
        ./configs/bin/yq e '.adversarial_training_method = "AutoAttack"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_4.$iter.yml
        sbatch submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_4.$iter.yml $iter
    done
done
