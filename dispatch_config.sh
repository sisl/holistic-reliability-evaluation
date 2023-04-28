#!/usr/bin/env bash
## This script reads a dhall config file (first argument) with possible modifications
## that returns a /List/ of configs.
## The dhall file is converted to yaml and then split into one file per config, which
## are stored in a unique directory.
## Then a batch job is launched for each of those files.
set -e
shopt -s expand_aliases
alias sbatch=echo

# make a unique directory where the config yamls are stored
TMPDIR=$(mktemp -d -p configs)
echo $TMPDIR
./configs/bin/dhall-to-yaml-ng --file $1 > $TMPDIR/config.yaml

# split the one yaml into a set of individual config files, and submit them as jobs!
LEN=$(./configs/bin/yq 'length' $TMPDIR/config.yaml)
for i in $(seq 0 $(($LEN - 1))); do
    ./configs/bin/yq ".[$i]" $TMPDIR/config.yaml > $TMPDIR/"run$i.yaml"
    sbatch submit_config.sh $TMPDIR/"run$i.yaml"
done
