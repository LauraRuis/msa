#!/bin/bash

export DATA_DIRECTORY=$1
export SEED=$2
export MODE=$3
export MODEL_TYPE=$4
export SIMPLIFIED_ARCHITECTURE=$5
export USE_ATTENTION=$6
export USE_CONDITIONAL_ATTENTION=$7
export TYPE_ATTENTION=$8
export ATTENTION_KEY=$9
export CONDITIONAL_ATTENTION_KEY=${10}
export UPSAMPLE_ISOLATED=${11}
export WEIGHT_DECAY=${12}
export COLLAPSE_ALO=${13}
export MODULAR=${14}
export K=${15}
export MODEL_SIZE=${16}
export ISOLATE_ADVERB_TYPES=${17}
export ONLY_KEEP_ADVERBS=${18}
export OUTPUT_DIRECTORY=${MODEL_TYPE}_modular_seed_${SEED}
export EXPERIMENT_NAME=${MODEL_TYPE}_modular_seed_${SEED}
export OUTPUT_FOLDER_PATTERN="%s_modular_seed_${SEED}"

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "${14}" ]; then
    echo "DATA_DIRECTORY: ${1}"
    echo "SEED: ${2}"
    echo "MODE: ${3}"
    echo "MODEL_TYPE: ${4}"
    echo "SIMPLIFIED_ARCHITECTURE: ${5}"
    echo "MODULAR: ${14}"
    echo Invalid Arguments
    exit 1
fi

mkdir ${OUTPUT_DIRECTORY}
echo "Output directory: ${OUTPUT_DIRECTORY}"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Seed: ${SEED}"
echo "Mode: ${MODE} (options: train, test, error_analysis)"
echo "Model type: ${MODEL_TYPE}"
echo "Simplified architecture: ${SIMPLIFIED_ARCHITECTURE}"
echo "Use attention: ${USE_ATTENTION}"
echo "Use conditional attention: ${USE_CONDITIONAL_ATTENTION}"
echo "Type attention: ${TYPE_ATTENTION}"
echo "Attention key: ${ATTENTION_KEY}"
echo "Conditional attention key: ${CONDITIONAL_ATTENTION_KEY}"
echo "Upsample isolated: ${UPSAMPLE_ISOLATED}"
echo "Weight decay: ${WEIGHT_DECAY}"
echo "Output folder pattern: ${OUTPUT_FOLDER_PATTERN}"
echo "Modular: ${MODULAR}"
echo "K: ${K}"
echo "Isolate adverb types: ${ISOLATE_ADVERB_TYPES}"
echo "Only keep adverbs: ${ONLY_KEEP_ADVERBS}"


if [ "${MODE}" == "train" ]; then
  sbatch --job-name=train_${EXPERIMENT_NAME} --output=${EXPERIMENT_NAME}.out --error=${EXPERIMENT_NAME}.err \
   slurm_adverb_train_modular.sh ${DATA_DIRECTORY} ${OUTPUT_DIRECTORY} ${SEED} ${EXPERIMENT_NAME} ${MODEL_TYPE} ${SIMPLIFIED_ARCHITECTURE} \
   ${USE_ATTENTION} $USE_CONDITIONAL_ATTENTION $TYPE_ATTENTION $ATTENTION_KEY $CONDITIONAL_ATTENTION_KEY ${UPSAMPLE_ISOLATED} ${WEIGHT_DECAY} \
   ${COLLAPSE_ALO} ${K} ${MODEL_SIZE} ${ISOLATE_ADVERB_TYPES} ${ONLY_KEEP_ADVERBS}
fi
if [ "${MODE}" == "test" ]; then
  sbatch --job-name=test_${EXPERIMENT_NAME} --output=${EXPERIMENT_NAME}.out --error=${EXPERIMENT_NAME}.err \
   slurm_adverb_test_modular.sh ${DATA_DIRECTORY} ${OUTPUT_DIRECTORY} ${EXPERIMENT_NAME} ${MODEL_TYPE} ${SIMPLIFIED_ARCHITECTURE} \
   ${USE_ATTENTION} $USE_CONDITIONAL_ATTENTION $TYPE_ATTENTION $ATTENTION_KEY $CONDITIONAL_ATTENTION_KEY ${OUTPUT_FOLDER_PATTERN} \
    ${MODEL_SIZE} ${COLLAPSE_ALO} ${MODULAR}
fi
if [ "${MODE}" == "error_analysis" ]; then
  sbatch --job-name=err_${EXPERIMENT_NAME} --output=${EXPERIMENT_NAME}.out --error=${EXPERIMENT_NAME}.err \
   slurm_adverb_error_analysis_modular.sh ${DATA_DIRECTORY} ${OUTPUT_DIRECTORY} ${EXPERIMENT_NAME}
fi
echo "Done."
