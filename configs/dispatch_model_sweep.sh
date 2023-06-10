#!/usr/bin/env sh

DEFAULT_CONFIG='iwildcam-defaults.yml'
MODEL_SOURCES=('torchvision' 'open_clip' 'mae')
TMPD="tmp2"
YQ_TMP=$(mktemp)

for iter in {1..1}; do
    for MODEL_SOURCE in "${MODEL_SOURCES[@]}"; do
        YQ_IN=".save_folder = \"/scratch/users/romeov/holistic-reliability-evaluation/results\" |
               .data_dir = \"/scratch/users/romeov/holistic-reliability-evaluation/data\" |
               .max_num_workers = 16 |
               .batch_size = .batch_size / 2 | .batch_size tag=\"!!int\" |
               .adversarial_training_method = \"FGSM\" |
               .adversarial_training_eps = \"3/255\" |
               .eval_transforms = [\"wilds_default_normalization\"] |
               .algorithm = \"model_sweep_adversarial\" |
               .model_source = \"${MODEL_SOURCE}\""
        echo $YQ_IN > $YQ_TMP
        ./configs/bin/yq --from-file $YQ_TMP configs/$DEFAULT_CONFIG > $TMPD/$DEFAULT_CONFIG
        # sbatch submit_config.sh $TMPD/$DEFAULT_CONFIG

        # ./configs/bin/yq e '.adversarial_training_eps = "1/255"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_${MODEL_SOURCE}_1.$iter.yml
        # sbatch submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_1.$iter.yml $iter
        # ./configs/bin/yq e '.adversarial_training_eps = "8/255"' $TMPD/$DEFAULT_CONFIG > $TMPD/${DEFAULT_CONFIG%.*}_${MODEL_SOURCE}_2.$iter.yml
        # sbatch submit_config.sh $TMPD/${DEFAULT_CONFIG%.*}_2.$iter.yml $iter
    done
done
