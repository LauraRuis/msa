#!/bin/bash

export DATA_DIRECTORY=$1
export MODEL_TYPE=regular
export NUM_EXTRA_ADVERBS=$2
export SEED=$3
export K=$4
export MODE=$5
export MODEL_SIZE=$6
export OUTPUT_DIRECTORY=${MODEL_TYPE}_model_${NUM_EXTRA_ADVERBS}_extra_adverbs_${K}_shot_${MODEL_SIZE}_model_seed_${SEED}
export EXPERIMENT_NAME=${MODEL_TYPE}_model_${NUM_EXTRA_ADVERBS}_extra_adverbs_${K}_shot_${MODEL_SIZE}_model_seed_${SEED}

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ];then
    echo "DATA_DIRECTORY: ${1}"
    echo "NUM_EXTRA_ADVERBS: ${2}"
    echo "SEED: ${3}"
    echo "K-SHOT LEARNING OF CAUTIOUSLY ETC.: ${4}"
    echo "MODE: ${5}"
    echo "MODEL_SIZE: ${6}"
    echo Invalid Arguments
    exit 1
fi

mkdir ${OUTPUT_DIRECTORY}
echo "Output directory: ${OUTPUT_DIRECTORY}"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Seed: ${SEED}"
echo "K-shot learning of cautiously etc.: ${K}"
echo "Mode: ${MODE} (options: train, test, error_analysis)"
echo "Model size: ${MODEL_SIZE} (options: small, big)"

if [ "${MODE}" == "train" ]; then
  sbatch --job-name=train_${EXPERIMENT_NAME} --output=${EXPERIMENT_NAME}.out --error=${EXPERIMENT_NAME}.err \
   slurm_adverb_train_regular.sh ${DATA_DIRECTORY} ${OUTPUT_DIRECTORY} ${SEED} ${EXPERIMENT_NAME} ${MODEL_SIZE}
fi
if [ "${MODE}" == "test" ]; then
  sbatch --job-name=test_${EXPERIMENT_NAME} --output=${EXPERIMENT_NAME}.out --error=${EXPERIMENT_NAME}.err \
   slurm_adverb_test_regular.sh ${DATA_DIRECTORY} ${OUTPUT_DIRECTORY} ${EXPERIMENT_NAME} ${MODEL_SIZE}
fi
if [ "${MODE}" == "error_analysis" ]; then
  sbatch --job-name=err_${EXPERIMENT_NAME} --output=${EXPERIMENT_NAME}.out --error=${EXPERIMENT_NAME}.err \
   slurm_adverb_error_analysis_regular.sh ${DATA_DIRECTORY} ${OUTPUT_DIRECTORY} ${EXPERIMENT_NAME}
fi
echo "Done."