#!/usr/bin/env bash
## This script reads a dhall config file (first argument) with possible modifications
## that returns a /List/ of configs.
## The dhall file is converted to yaml and then split into one file per config, which
## are stored in a unique directory.
## Then we can forward the configs and lauch a batch job for each file, see example usage.

# Example Usage:
# `bash dispatch_config.sh configs/mod0.dhall  | xargs -n1 sbatch submit_config.sh`

set -e

# make a unique directory where the config yamls are stored
DIR=$(mktemp -d -p configs)
./configs/bin/dhall-to-yaml-ng --documents --file $1 > $DIR/config.yaml
./configs/bin/yq --no-doc --split-exp '"run_"+$index' $DIR/config.yaml
mv run*.yml $DIR

ls -1 $DIR/run* | cat
