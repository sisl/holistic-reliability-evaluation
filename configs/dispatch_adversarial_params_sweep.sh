#!/usr/bin/env bash

# for debugging purposes
if ! command -v sbatch &> /dev/null
then
    echo "Command <sbatch> not found. Probably, you are not on the cluster. Replacing with <echo> for debugging."
    shopt -s expand_aliases
    alias sbatch=echo
fi

DEFAULT_CONFIGS=('iwildcam-defaults.yml' 'fmow-defaults.yml' 'camelyon17-defaults.yml' 'rxrx1-defaults.yml')

# We stash copies of the derived config files and submit them all individually.
# Recall that it may be some time (~hours) until the config file is read, so we must not modify it until then!
STASH_DIR="configs/.adversarial_config_stash/attack_params_sweep"
mkdir -p $STASH_DIR

export ALGO_NAME
export BASELINE_ALGO_NAME
export SAVE_DIR=${SCRATCH}/holistic-reliability-evaluation/results
export DATA_DIR=${GROUP_SCRATCH}/holistic-reliability-evaluation/data

for DEFAULT_CONFIG in "${DEFAULT_CONFIGS[@]}"; do
    ADV_BASELINE_CFG=${DEFAULT_CONFIG%.*}_adversarial-baseline-cfg.yml
    BASELINE_ALGO_NAME="adversarial_training_ResNet-method=PGD-eps=3_255"
    ./configs/bin/yq '.save_folder = env(SAVE_DIR) |
                      .data_dir    = env(DATA_DIR) |
                      .max_num_workers = 16 |
                      .batch_size = .batch_size / 2 | .batch_size tag="!!int" |
                      .adversarial_training_method = "PGD" |
                      .adversarial_training_eps = "3/255" |
                      .eval_transforms = ["wilds_default_normalization"] |
                      .algorithm = env(BASELINE_ALGO_NAME)' configs/$DEFAULT_CONFIG \
            > $STASH_DIR/$ADV_BASELINE_CFG

    # for camelyon we have to half the batch size one more time.
    if [[ "$DEFAULT_CONFIG" = "camelyon17-defaults.yml" ]]; then
        # notice this is in place (-i)
        ./configs/bin/yq -i '.batch_size = .batch_size / 2 | .batch_size tag="!!int"' $STASH_DIR/$ADV_BASELINE_CFG
    fi

    for seed in {1..2}; do  # the seed is passed to the `configs/submit_adv_config.sh` script, which passes it to pytorch-lightning
        cat $STASH_DIR/$ADV_BASELINE_CFG > $STASH_DIR/cfg_0.yml
        sbatch configs/submit_adv_config.sh.sh $STASH_DIR/cfg_0.yml $seed

        ALGO_NAME="adversarial_training_ResNet-method=PGD-eps=1_255"
        ./configs/bin/yq '.adversarial_training_eps = "1/255" |
                          .algorithm = env(ALGO_NAME)' \
              $STASH_DIR/$ADV_BASELINE_CFG > $STASH_DIR/cfg_1.yml
        sbatch configs/submit_adv_config.sh.sh $STASH_DIR/cfg_1.yml $seed

        ALGO_NAME="adversarial_training_ResNet-method=PGD-eps=8_255"
        ./configs/bin/yq '.adversarial_training_eps = "8/255" |
                          .algorithm = env(ALGO_NAME)' \
              $STASH_DIR/$ADV_BASELINE_CFG > $STASH_DIR/cfg_2.yml
        sbatch configs/submit_adv_config.sh.sh $STASH_DIR/cfg_2.yml $seed

        ALGO_NAME="adversarial_training_ResNet-method=FGSM-eps=3_255"
        ./configs/bin/yq '.adversarial_training_method = "FGSM" |
                          .algorithm = env(ALGO_NAME)' \
              $STASH_DIR/$ADV_BASELINE_CFG > $STASH_DIR/cfg_3.yml
        sbatch configs/submit_adv_config.sh.sh $STASH_DIR/cfg_3.yml $seed

        ALGO_NAME="adversarial_training_ResNet-method=AutoAttack-eps=3_255"
        ./configs/bin/yq '.adversarial_training_method = "AutoAttack" |
                          .algorithm = env(ALGO_NAME)' \
              $STASH_DIR/$ADV_BASELINE_CFG > $STASH_DIR/cfg_4.yml
        sbatch configs/submit_adv_config.sh.sh $STASH_DIR/cfg_4.yml $seed
    done
done
