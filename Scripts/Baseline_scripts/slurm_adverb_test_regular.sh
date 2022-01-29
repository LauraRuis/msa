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
echo "Starting Job.."
source gscan_seq2seq_new/bin/activate
srun python3.8 -m seq2seq --mode=test --data_directory=data/${1} --attention_type=bahdanau --no_auxiliary_task \
  --conditional_attention --output_directory=${2} --resume_from_file=${2}/model_best.pth.tar \
  --splits=test,adverb_1,adverb_2,contextual,situational_1,situational_2,visual,visual_easier --model_size=${4}\
  --output_file_name=${2}_predict.json --max_decoding_steps=120 &> ${2}/${3}_test.txt
