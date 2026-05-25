#!/bin/bash
#SBATCH --job-name=p3l
#SBATCH --account=csd403
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 --mem=128G
#SBATCH --time=02:30:00
#SBATCH --output=/expanse/lustre/projects/csd403/bpinto/neuralbench_save/p3l_%x_%j.out

set -euo pipefail
source /expanse/lustre/projects/csd403/bpinto/envs/neuroai/bin/activate
cd /expanse/lustre/projects/csd403/bpinto/neuroai/neuralbench-repo
# CRITICAL: all locks on /scratch local NVMe, not Lustre, to avoid ESTALE
export MNE_CONFIG_DIR=$SLURM_TMPDIR/mne_cfg
export MNE_DATA=$SLURM_TMPDIR/mne_data
export MOABB_DATA=$SLURM_TMPDIR/moabb_data
mkdir -p "$MNE_CONFIG_DIR" "$MNE_DATA" "$MOABB_DATA"
# Pre-create MNE config so concurrent workers don't race on creation
echo '{}' > "$MNE_CONFIG_DIR/mne-python.json"
# Skip MNE config lookup for STIM channel — set the default directly
export MNE_STIM_CHANNEL="STI 014"
export WANDB_MODE=disabled
export MOABB_ACCEPT_LICENCE=1

# Map DS short tag to actual dataset directory name
case "$DS" in
    chavarriaga2010) DSDIR="Chavarriaga2010Learning" ;;
    shin2017b)       DSDIR="Shin2017OpenB" ;;
    hinss2021)       DSDIR="Hinss2021Open" ;;
    ofner2017)       DSDIR="Ofner2017UpperExecution" ;;
    kappenman2021p3) DSDIR="Kappenman2021ErpP3" ;;
    tangermann2012)  DSDIR="Tangermann2012Review" ;;
    reichert2020)    DSDIR="Reichert2020Impact" ;;
    thielen2015)     DSDIR="Thielen2015Broad" ;;
    lee2019ssvep)    DSDIR="Lee2019EegSsvep" ;;
    *) echo "unknown DS=$DS"; exit 2 ;;
esac

SRC=/expanse/lustre/projects/csd403/bpinto/data/moabb/$DSDIR
DST=$SLURM_TMPDIR/moabb/$DSDIR
echo "host: $(hostname)  FM=$FM TASK=$TASK DS=$DS  DSDIR=$DSDIR"
echo "rsync $SRC → $DST"

mkdir -p "$DST"
# rsync the entire dataset (download/ + timelines.csv etc.) to local scratch
time rsync -a "$SRC/" "$DST/"
du -sh "$DST"

export SLURM_DATA_PATH="$DST"
echo "SLURM_DATA_PATH=$SLURM_DATA_PATH"

python run_phase3.py \
    --model "$FM" --task "$TASK" --dataset "$DS" \
    --probes-file "$PROBES_FILE" \
    --aggregation flatten \
    --seeds 33,34,35 \
    --experiments-per-job all \
    --local-workers-per-job 1
echo "exit=$?"
