#!/bin/bash

# All commands that start with SBATCH contain commands that are just
# used by SLURM for scheduling

# Time you think you need; default is one hour.
# In this case, hh:mm:ss,
# The less you ask for the faster your job will run.
#SBATCH --time=48:00:00

# --gres will give you one GPU, you can ask for more, up to 8 (or how ever many are on the node/card)
# you can also ask for different types of GPU with a larger memory, but the job might queue until the GPU is free
# e.g.,: gpu:v100:1
#SBATCH --gres gpu:1

# We are submitting to the gpu partition, if you can submit to the hns partition, change this to -p hns_gpu.


# Number of nodes you are requesting.
#SBATCH --nodes=1

# memory per node; default is 4000 MB per CPU (e.g., 180GB)
#SBATCH --mem=200GB

# Get an email if job ends or fails.
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gr069
# Choose which node to submit the job to, can also leave this out and let the scheduler choose.

echo "Starting Job.."
source modular/bin/activate
srun python3.8 -m Modularity --mode=train --data_directory=data/${1} --output_directory=${2} --seed=${3} \
      --generate_vocabularies --max_decoding_steps=120 --module=${5} --max_training_iterations=200000 \
      --simplified_architecture=${6} --use_attention=${7} --use_conditional_attention=${8} --upsample_isolated=${12} \
      --type_attention=${9} --attention_values_key=${10} --conditional_attention_values_key=${11} --training_batch_size=200 \
      --weight_decay=${13} --collapse_alo=${14} --k=${15} --isolate_adverb_types=${17} --only_keep_adverbs=${18} \
      --model_size=${16} &> ${2}/train_${4}.txt
