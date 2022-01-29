#!/usr/bin/env bash

DATA_DIRECTORY=$1
NUM_EXTRA_ADVERBS=$2
SEEDS=$3
K=$4
MODE=$5
MODEL_SIZE=$6

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "DATA_DIRECTORY: ${1}"
    echo "NUM_EXTRA_ADVERBS: ${2}"
    echo "SEEDS: ${3}"
    echo "K-SHOT LEARNING OF CAUTIOUSLY ETC.: ${4}"
    echo "MODE: ${5}"
    echo Invalid Arguments
    exit 1
fi

chmod a+x experiment_call_script.sh

seeds=$(echo $SEEDS | tr "," "\n")

for seed in $seeds
do
    ./experiment_call_script.sh $DATA_DIRECTORY $NUM_EXTRA_ADVERBS $seed $K $MODE $MODEL_SIZE
done