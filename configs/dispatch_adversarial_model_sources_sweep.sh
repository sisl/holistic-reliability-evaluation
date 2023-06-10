#!/usr/bin/env bash

# for debugging purposes
if ! command -v sbatch &> /dev/null
then
    echo "Command <sbatch> not found. Probably, you are not on the cluster. Replacing with <echo> for debugging."
    shopt -s expand_aliases
    alias sbatch=echo
fi

DEFAULT_CONFIG='iwildcam-finetune.yml'
MODEL_SOURCES=('torchvision' 'open_clip' 'mae')

# We stash copies of the derived config files and submit them all individually.
# Recall that it may be some time (~hours) until the config file is read, so we must not modify it until then!
STASH_DIR="configs/.adversarial_config_stash/model_sources_sweep"
mkdir -p $STASH_DIR

export SAVE_DIR=${SCRATCH}/holistic-reliability-evaluation/results
export DATA_DIR=${GROUP_SCRATCH}/holistic-reliability-evaluation/data
./configs/bin/yq '.save_folder = env(SAVE_DIR) |
                  .data_dir    = env(DATA_DIR) |
                  .max_num_workers = 16 |
                  .batch_size = .batch_size / 2 | .batch_size tag="!!int" |
                  .adversarial_training_method = "FGSM" |
                  .adversarial_training_eps = "3/255" |
                  .eval_transforms = ["wilds_default_normalization"]' configs/$DEFAULT_CONFIG\
        > $STASH_DIR/adversarial-config-base.yml


for seed in {1..2}; do  # the seed is passed to the `submit_config` script, which passes it to pytorch-lightning
    for MODEL_SOURCE in "${MODEL_SOURCES[@]}"; do
        # we export these variables for use in yq
        export ALGO_NAME="adversarial_training_ViT-model_source_${MODEL_SOURCE}"
        export MODEL_SOURCE
        ./configs/bin/yq  '.algorithm    = env(ALGO_NAME) |
                           .model_source = env(MODEL_SOURCE)' $STASH_DIR/adversarial-config-base.yml \
                > $STASH_DIR/adversarial-config--model-source=${MODEL_SOURCE}.yml
        sbatch configs/submit_adv_config.sh $STASH_DIR/adversarial-config--model-source=${MODEL_SOURCE}.yml $seed
    done
done
