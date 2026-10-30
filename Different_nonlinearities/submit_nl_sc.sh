#!/bin/bash -l

#SBATCH -o ./logs/gradnlsc.%A_%a.out
#SBATCH -e ./logs/gradnlsc.%A_%a.err
#SBATCH -D ./
#SBATCH -J gradnl
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --mail-type=none
#SBATCH --mail-user=nicola.dalcin@mpl.mpg.de
# #SBATCH --exclude=...   # (site-specific node exclusions removed for release; adapt to your cluster)

# "Other nonlinearities" sweep, SEMICLASSICAL convention (sc_core.py physics).
# Cluster files: sc_core.py (parent dir or alongside), gradient_worker_nl_sc.py. — SB trains XOR for saturable & TPA nonlinearities.
# Fixed nudge ratio (RATIO_FIXED=0.04 in gradient_worker_nl_sc.py). No ratio sweep.
# Single SB mode for BOTH noise settings (set MODES below after the smoke test).
#
# Grid: 3 nl × 2 noise × 4 seeds × 1 mode = 24 jobs.  (drop 'kerr' for 16.)
#
#   mkdir -p logs results_nl_sc
#   sbatch --array=0-23 submit_nl_sc.sh

# Activate your Python environment here, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate sb-env
conda activate jax-env

# ── Parameter grid (must match gradient_worker_nl_sc.py) ────────────────────────
NL_VALUES=(kerr saturable tpa)
GA0_VALUES=(3.0 3.0 5.0)          # per-nl ga0 (index-aligned with NL_VALUES); tpa uses γ/g=0.3
NOISE_VALUES=(0.0 1.0)
SEEDS=(0 1 2 3)
MODES=(sigma_y)                    # <-- single mode; set from smoke_nl result

n_nl=${#NL_VALUES[@]}
n_noise=${#NOISE_VALUES[@]}
n_seed=${#SEEDS[@]}
n_mode=${#MODES[@]}
total_jobs=$(( n_nl * n_noise * n_seed * n_mode ))

OFFSET=${OFFSET:-0}
IDX=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
if [[ $IDX -ge $total_jobs ]]; then
    echo "Error: IDX=$IDX >= total_jobs=$total_jobs"; exit 1
fi

# ── Decode flat index → (nl, noise, seed, mode) ──────────────────────────────
mode_idx=$((  IDX % n_mode  )); IDX=$(( IDX / n_mode  ))
seed_idx=$((  IDX % n_seed  )); IDX=$(( IDX / n_seed  ))
noise_idx=$(( IDX % n_noise )); IDX=$(( IDX / n_noise ))
nl_idx=$((    IDX % n_nl    ))

NL=${NL_VALUES[$nl_idx]}
GA0=${GA0_VALUES[$nl_idx]}
NOISE=${NOISE_VALUES[$noise_idx]}
SEED=${SEEDS[$seed_idx]}
MODE=${MODES[$mode_idx]}

echo "Job $((SLURM_ARRAY_TASK_ID + OFFSET)) / $total_jobs: nl=$NL ga0=$GA0 noise=$NOISE seed=$SEED mode=$MODE"

srun python gradient_worker_nl_sc.py \
    --nl          $NL    \
    --ga0         $GA0   \
    --noise_scale $NOISE \
    --seed        $SEED  \
    --mode        $MODE
