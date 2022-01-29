# Experiment Scripts

This folder contains all scripts to run the experiments in the paper. 

## Run locally
For both the baseline and the modular model the commands to run directly on your machine are in `all_experiments.sh` in the respective folders.

## Run on cluster
For both the baseline and the modular model the commands to run on some cluster with slurm can be used as follows:

### Modular Experiments on cluster
Train each module separately by changing the parameters as you want them in `start_<module>.sh` (e.g., `start_interaction.sh` for the interaction module).
Run the script.

### Baseline Experiments on cluster
Change the parameters as you want them in `start_experiments.sh` and run the script.
