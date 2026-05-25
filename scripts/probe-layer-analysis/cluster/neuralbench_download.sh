#!/bin/bash
#SBATCH --job-name=nbdl
#SBATCH --account=csd403
#SBATCH --partition=shared
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/expanse/lustre/projects/csd403/bpinto/neuralbench_save/nbdl_%x_%j.out

source /expanse/lustre/projects/csd403/bpinto/envs/neuroai/bin/activate
cd /expanse/lustre/projects/csd403/bpinto/neuroai/neuralbench-repo
export MNE_CONFIG_DIR=/expanse/lustre/projects/csd403/bpinto/mne_cfg/${SLURM_JOB_ID}
mkdir -p "$MNE_CONFIG_DIR"
export WANDB_MODE=disabled
export MOABB_ACCEPT_LICENCE=1

echo "host: $(hostname)  TASK=$TASK  DS=$DS"

# Use the actual neuralbench CLI with --download flag
neuralbench eeg "$TASK" --download --dataset "$DS"
echo "exit=$?"
