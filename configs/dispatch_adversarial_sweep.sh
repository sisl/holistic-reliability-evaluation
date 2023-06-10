#!/usr/bin/env sh

DEFAULT_CONFIGS=('iwildcam-defaults.yml' 'fmow-defaults.yml' 'camelyon17-defaults.yml' 'rxrx1-defaults.yml')
TMPD="tmp"

for iter in {2..2}; do
    for DEFAULT_CONFIG in "${DEFAULT_CONFIGS[@]}"; do
        ./configs/bin/yq '.save_folder = "/scratch/users/romeov/holistic-reliability-evaluation/results" |
                          .data_dir = "/scratch/users/romeov/holistic-reliability-evaluation/data" |
                          .max_num_workers = 16 |
                          .batch_size = .batch_size / 2 | .batch_size tag="!!int" |
                          .adversarial_training_method = "PGD" |
                          .adversarial_training_eps = "3/255" |
                          .eval_transforms = ["wilds_default_normalization"] |
                          .algorithm = "adversarial_sweep"'\
                           configs/$DEFAULT_CONFIG > $TMPD/$DEFAULT_CONFIG
        if [[ "$DEFAULT_CONFIG" = "camelyon17-defaults.yml" ]]; then  # for camelyon we have to half one more time.
            ./configs/bin/yq -i '.batch_size = .batch_size / 2 | .batch_size tag="!!int"' $TMPD/$DEFAULT_CONFIG
        fi
        sbatch submit_config.sh $TMPD/$DEFAULT_CONFIG

        ./configs/bin/yq '.adversarial_training_eps = "1/255"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_1.$iter.yml
        sbatch submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_1.$iter.yml $iter
        ./configs/bin/yq '.adversarial_training_eps = "8/255"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_2.$iter.yml
        sbatch submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_2.$iter.yml $iter


        ./configs/bin/yq '.adversarial_training_method = "FGSM"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_3.$iter.yml
        sbatch submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_3.$iter.yml $iter
        ./configs/bin/yq '.adversarial_training_method = "AutoAttack" |
                          .batch_size = .batch_size / 2 | .batch_size tag="!!int"'\
                          $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_4.$iter.yml
        sbatch submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_4.$iter.yml $iter
    done
done