#!/usr/bin/env bash

DATA_DIRECTORY=$1
SEEDS=$2
MODE=$3
MODEL_TYPE=$4
SIMPLIFIED_ARCHITECTURE=$5
USE_ATTENTION=$6
USE_CONDITIONAL_ATTENTION=$7
TYPE_ATTENTION=$8
ATTENTION_KEY=$9
CONDITIONAL_ATTENTION_KEY=${10}
UPSAMPLE_ISOLATED=${11}
WEIGHT_DECAY=${12}
COLLAPSE_ALO=${13}
MODULAR=${14}
K=${15}
MODEL_SIZE=${16}
ISOLATE_ADVERB_TYPES=${17}
ONLY_KEEP_ADVERBS=${18}


if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "${14}" ] || [ -z "${16}" ]; then
    echo "DATA_DIRECTORY: ${1}"
    echo "SEEDS: ${2}"
    echo "MODE: ${3}"
    echo "MODEL_TYPE: ${4}"
    echo "SIMPLIFIED_ARCHITECTURE: ${5}"
    echo "MODULAR: ${14}"
    echo "MODEL_SIZE: ${16}"
    echo Invalid Arguments
    exit 1
fi

chmod a+x experiment_call_script.sh

seeds=$(echo $SEEDS | tr "," "\n")

for seed in $seeds
do
    ./experiment_call_script.sh $DATA_DIRECTORY $seed $MODE $MODEL_TYPE $SIMPLIFIED_ARCHITECTURE $USE_ATTENTION $USE_CONDITIONAL_ATTENTION $TYPE_ATTENTION $ATTENTION_KEY $CONDITIONAL_ATTENTION_KEY ${UPSAMPLE_ISOLATED} ${WEIGHT_DECAY} ${COLLAPSE_ALO} ${MODULAR} ${K} ${MODEL_SIZE} ${ISOLATE_ADVERB_TYPES} ${ONLY_KEEP_ADVERBS}
done
