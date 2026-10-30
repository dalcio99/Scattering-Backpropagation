#!/bin/bash -l

#SBATCH -o ./logs/gradsct.%A_%a.out
#SBATCH -e ./logs/gradsct.%A_%a.err
#SBATCH -D ./
#SBATCH -J gradsct
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

# Semiclassical-convention T_m (measurement-time) study.
# Mirror of the sweep-20 tsamp study with sc_core.py physics.
# Fixed: ga0=1.0, ratio=0.05, noise_scale=1.0, lr=5e-4, 500 epochs.
# T_m ∈ {25, 50, 100, 200, 400}; loss evaluated at <y>_{T_m} (record-based).
# Files needed on the cluster: sc_core.py, gradient_worker_sc.py,
# gradient_worker_sc_tsamp.py, this script.
#
# Grid: 5 T_m × 8 seeds × 2 modes = 80 jobs
#
#   mkdir -p logs results_tsamp_sc
#   sbatch --array=0-79 submit_sc_tsamp.sh

# Activate your Python environment here, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate sb-env
conda activate jax-env

# ── Parameter grid ────────────────────────────────────────────────────────────
T_SAMP_VALUES=(25 50 100 200 400)
SEEDS=(0 1 2 3 4 5 6 7)
MODES=(sigma_x sigma_y)

n_tsamp=${#T_SAMP_VALUES[@]}
n_seed=${#SEEDS[@]}
n_mode=${#MODES[@]}

total_jobs=$(( n_tsamp * n_seed * n_mode ))

IDX=$SLURM_ARRAY_TASK_ID
if [[ $IDX -ge $total_jobs ]]; then
    echo "Error: IDX=$IDX >= total_jobs=$total_jobs"
    exit 1
fi

# ── Decode flat index → (t_samp, seed, mode) ─────────────────────────────────
mode_idx=$(( IDX % n_mode  ));  IDX=$(( IDX / n_mode  ))
seed_idx=$(( IDX % n_seed  ));  IDX=$(( IDX / n_seed  ))
tsamp_idx=$(( IDX % n_tsamp ))

T_SAMP=${T_SAMP_VALUES[$tsamp_idx]}
SEED=${SEEDS[$seed_idx]}
MODE=${MODES[$mode_idx]}

echo "Job $SLURM_ARRAY_TASK_ID / $total_jobs: mode=$MODE t_sample=$T_SAMP seed=$SEED"

srun python gradient_worker_sc_tsamp.py \
    --t_sample $T_SAMP \
    --seed     $SEED   \
    --mode     $MODE
