#!/bin/bash
# Full v0 core sweep: 6 FMs × 10-11 probe layers × 3 seeds × 9 tasks.
# One sbatch job per (FM, task) — packed into local job runs ~30 experiments.
# Each job rsyncs its dataset to /scratch ($SLURM_TMPDIR/moabb/<Dataset>) and
# patches the source.path so neither Lustre nor cross-job races can hit.

LOG=/tmp/full_sweep.log
exec > "$LOG" 2>&1
PROBES_DIR=/expanse/lustre/projects/csd403/bpinto/neuralbench_save
MAX=22

declare -a JOBS=()
for FM in bendr biot cbramod labram luna reve; do
    for TD in \
        "cvep,thielen2015" "ern,chavarriaga2010" "mental_arithmetic,shin2017b" \
        "mental_workload,hinss2021" "motor_execution,ofner2017" \
        "motor_imagery,tangermann2012" "n2pc,reichert2020" \
        "p3,kappenman2021p3" "ssvep,lee2019ssvep"; do
        JOBS+=("$FM,${TD%,*},${TD#*,},$PROBES_DIR/probes3_${FM}_full.txt")
    done
done
echo "[$(date +%H:%M:%S)] full sweep: ${#JOBS[@]} jobs to submit"

i=0
while [ "$i" -lt "${#JOBS[@]}" ]; do
    CURRENT=$(squeue -u bpinto -h | wc -l)
    if [ "$CURRENT" -ge "$MAX" ]; then sleep 60; continue; fi
    CAN=$(( MAX - CURRENT ))
    for k in $(seq 1 "$CAN"); do
        if [ "$i" -ge "${#JOBS[@]}" ]; then break; fi
        IFS=, read -r FM TASK DS PF <<< "${JOBS[$i]}"
        JID=$(sbatch --parsable \
            --time=02:00:00 \
            --partition=gpu-shared \
            --export=ALL,FM=$FM,TASK=$TASK,DS=$DS,PROBES_FILE=$PF \
            --job-name=p3f_${FM}_${TASK} \
            /tmp/phase3_local.sh 2>/dev/null) || { sleep 10; break; }
        echo "  [$((i+1))/${#JOBS[@]}] $FM × $TASK → $JID"
        i=$((i+1))
    done
done
echo "[$(date +%H:%M:%S)] all submitted; waiting for completion"

# Wait for all p3f_ jobs
while true; do
    LEFT=$(squeue -u bpinto -h -o "%j" 2>/dev/null | grep -c "^p3f_")
    if [ "$LEFT" -eq 0 ]; then break; fi
    echo "  [$(date +%H:%M:%S)] $LEFT p3f jobs left"
    sleep 300
done

# Aggregate
echo "[$(date +%H:%M:%S)] running collect_all.py"
source /expanse/lustre/projects/csd403/bpinto/envs/neuroai/bin/activate
cd /expanse/lustre/projects/csd403/bpinto/neuroai/neuralbench-repo
python collect_all.py > /expanse/lustre/projects/csd403/bpinto/neuralbench_save/full_collect.out 2>&1
echo "[$(date +%H:%M:%S)] DONE"
