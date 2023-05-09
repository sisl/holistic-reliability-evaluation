#!/usr/bin/env sh

DEFAULT_CONFIGS=('iwildcam-defaults.yml' 'fmow-defaults.yml' 'camelyon17-defaults.yml' 'rxrx1-defaults.yml')
TMPD=$(mktemp -d)

for DEFAULT_CONFIG in "${DEFAULT_CONFIGS[@]}"; do
    for iter in {1..3}; do
        ./configs/bin/yq e '.adversarial_training_method = "PGD", .adversarial_training_eps = "3/255", .algorithm = "adversarial_sweep"' configs/$DEFAULT_CONFIG > $TMPD/$DEFAULT_CONFIG
        "bsub submit_config.sh $TMPD/$DEFAULT_CONFIG"

        ./configs/bin/yq e '.adversarial_training_eps = "1/255"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_1.$iter.yml
        bsub submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_1.$iter.yml
        ./configs/bin/yq e '.adversarial_training_eps = "8/255"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_2.$iter.yml
        bsub submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_2.$iter.yml


        ./configs/bin/yq e '.adversarial_training_method = "FGSM"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_3.$iter.yml
        bsub submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_3.$iter.yml
        ./configs/bin/yq e '.adversarial_training_method = "AutoAttack"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_4.$iter.yml
        bsub submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_4.$iter.yml
    done
done
