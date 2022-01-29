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

# Number of nodes you are requesting.
#SBATCH --nodes=1

# memory per node; default is 4000 MB per CPU (e.g., 180GB)
#SBATCH --mem=200GB

# Get an email if job ends or fails.
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gr069

# Choose which node to submit the job to, can also leave this out and let the scheduler choose.

module load python-3.8
module load cuda-10.1
module load cuda
echo "Starting Job.."
source gscan_seq2seq_new/bin/activate
srun python3.8 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=3500 \
  --data_directory=data/${1} --attention_type=bahdanau --no_auxiliary_task --generate_vocabularies \
  --conditional_attention --output_directory=${2} --training_batch_size=200 \
  --model_size=${5} --max_training_iterations=200000 --seed=${3} &> ${2}/${4}.txt
