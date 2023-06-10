#!/usr/bin/env sh

DEFAULT_CONFIGS=('iwildcam-defaults.yml' 'fmow-defaults.yml' 'camelyon17-defaults.yml' 'rxrx1-defaults.yml')

STASH_DIR="configs/.config_stash/attack_params_sweep"
mkdir -p $STASH_DIR

export ALGO_NAME
export BASELINE_ALGO_NAME

for DEFAULT_CONFIG in "${DEFAULT_CONFIGS[@]}"; do
    ADV_BASELINE_CFG=${DEFAULT_CONFIG%.*}_adversarial-baseline-cfg.yml
    BASELINE_ALGO_NAME="adversarial_training_ResNet-method=PGD-eps=3_255"
    ./configs/bin/yq '.save_folder =       env(SCRATCH)/holistic-reliability-evaluation/results |
                      .data_dir    = env(GROUP_SCRATCH)/holistic-reliability-evaluation/data |
                      .max_num_workers = 16 |
                      .batch_size = .batch_size / 2 | .batch_size tag=!!int |
                      .adversarial_training_method = PGD |
                      .adversarial_training_eps = 3/255 |
                      .eval_transforms = [wilds_default_normalization]
                      .algorithm = env(BASELINE_ALGO_NAME)' \
            configs/$DEFAULT_CONFIG \
            > $STASH_DIR/$ADV_BASELINE_CFG

    # for camelyon we have to half the batch size one more time.
    if [[ "$DEFAULT_CONFIG" = "camelyon17-defaults.yml" ]]; then
        ./configs/bin/yq -i '.batch_size = .batch_size / 2 | .batch_size tag=!!int' $STASH_DIR/${DEFAULT_CONFIG%.*}_adversarial-baseline-cfg.yml
    fi

    for seed in {1..2}; do  # the seed is passed to the `configs/submit_adv_config.sh` script, which passes it to pytorch-lightning
        cat $STASH_DIR/${DEFAULT_CONFIG%.*}_adversarial-baseline-cfg.yml > $STASH_DIR/cfg_0.yml
        sbatch configs/submit_adv_config.sh.sh $STASH_DIR/cfg_0.yml $seed

        ALGO_NAME="adversarial_training_ResNet-method=PGD-eps=1_255"
        ./configs/bin/yq '.adversarial_training_eps = 1/255 |
                          .algorithm = env(ALGO_NAME)' \
              $STASH_DIR/$ADV_BASELINE_CONFIG > $STASH_DIR/cfg_1.yml
        sbatch configs/submit_adv_config.sh.sh $STASH_DIR/cfg_1.yml $seed

        ALGO_NAME="adversarial_training_ResNet-method=PGD-eps=8_255"
        ./configs/bin/yq '.adversarial_training_eps = 8/255 |
                          .algorithm = env(ALGO_NAME)' \
              $STASH_DIR/$ADV_BASELINE_CONFIG > $STASH_DIR/cfg_2.yml
        sbatch configs/submit_adv_config.sh.sh $STASH_DIR/cfg_2.yml $seed

        ALGO_NAME="adversarial_training_ResNet-method=FGSM-eps=3_255"
        ./configs/bin/yq '.adversarial_training_method = FGSM |
                          .algorithm = env(ALGO_NAME)' \
              $STASH_DIR/$ADV_BASELINE_CONFIG > $STASH_DIR/cfg_3.yml
        sbatch configs/submit_adv_config.sh.sh $STASH_DIR/cfg_3.yml $seed

        ALGO_NAME="adversarial_training_ResNet-method=AutoAttack-eps=3_255"
        ./configs/bin/yq '.adversarial_training_method = AutoAttack |
                          .algorithm = env(ALGO_NAME)' \
              $STASH_DIR/$ADV_BASELINE_CONFIG > $STASH_DIR/cfg_4.yml
        sbatch configs/submit_adv_config.sh.sh $STASH_DIR/cfg_4.yml $seed
    done
done
