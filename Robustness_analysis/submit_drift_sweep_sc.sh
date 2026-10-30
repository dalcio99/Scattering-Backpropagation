#!/bin/bash -l

#SBATCH -o ./logs/drift21.%A_%a.out
#SBATCH -e ./logs/drift21.%A_%a.err
#SBATCH -D ./
#SBATCH -J driftsc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=02:00:00
#SBATCH --mail-type=none
#SBATCH --mail-user=nicola.dalcin@mpl.mpg.de
# #SBATCH --exclude=...   # (site-specific node exclusions removed for release; adapt to your cluster)

# -----------------------------------------------------------------------------------
# Drift-RATE sweep, SWEEP-21 conventions (matches robustness_seed_worker_sweep21
# and drift_worker_sweep21): a0_in_sq=160, fixed nudge ratio 0.04, lr=5e-4,
# train dt=0.05 / eval dt=0.025, T_trans=40, T_sample clean=20 / noisy=200,
# g = 1/320 hard-coded in the worker (g<|a_in|^2>/kbar^2 = 0.5). Same grid/mechanism 
# as submit_drift_sweep_ext.sh (extended low-end + trimmed top) so results are directly 
# comparable to the β=1 version.
#
# Embedded seed-0 networks come from the sweep-21 static pilot, so only
# drift_worker_sc.py is needed on the cluster.
#
# STAGE 0 (required): run robustness_seed_worker_sc.py --seed 0 first;
# the drift arms start from its trained J_clean/J_noisy networks.
# Grid: 15 sigma_step × 10 seeds × 3 arms = 450 jobs   (< 1000, single array)
# Results → results_drift_sweep_sc/ (does NOT touch the β=1 results dirs).
#
#   mkdir -p logs
#   sbatch --array=0-449 submit_drift_sweep_sc.sh
#
# Aggregate + plot when done (physical GHz top axis, Q=1e5):
#   conda run -n jax-env python plot_drift_sweep.py \
#       --results_dir results_drift_sweep_sc --Q 1e5
# ---------------------------------------------------------------------------

# Activate your Python environment here, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate sb-env
conda activate jax-env
# ---------------------------------------------------------------------------
#  Parameter grid (identical to submit_drift_sweep_ext.sh) 
# ---------------------------------------------------------------------------
SIGMA_STEPS=(0.000100 0.000177 0.000314 0.000556 0.000985 0.001745 0.003092 0.005477 0.009703 0.017191 0.030455 0.053954 0.095585 0.169338 0.300000)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
ARMS=(noisy clean frozen)

n_sigma=${#SIGMA_STEPS[@]}
n_seed=${#SEEDS[@]}
n_arm=${#ARMS[@]}
total_jobs=$(( n_sigma * n_seed * n_arm ))

OFFSET=${OFFSET:-0}
IDX=$(( SLURM_ARRAY_TASK_ID + OFFSET ))

if [ "$IDX" -ge "$total_jobs" ]; then
    echo "IDX $IDX >= total_jobs $total_jobs — nothing to do."
    exit 0
fi
# ---------------------------------------------------------------------------
#  Decompose flat index → (arm, sigma_step, seed) 
# ---------------------------------------------------------------------------
arm_i=$(( IDX / (n_sigma * n_seed) ))
rem=$(( IDX % (n_sigma * n_seed) ))
sigma_i=$(( rem / n_seed ))
seed_i=$(( rem % n_seed ))

ARM=${ARMS[$arm_i]}
SIGMA_STEP=${SIGMA_STEPS[$sigma_i]}
SEED=${SEEDS[$seed_i]}

echo "job $IDX / $total_jobs :  arm=$ARM  sigma_step=$SIGMA_STEP  seed=$SEED"

python drift_worker_sc.py \
    --arm "$ARM" \
    --sigma_step "$SIGMA_STEP" \
    --seed "$SEED" \
    --num_epochs 200 \
    --drift_mode common \
    --npz results_robustness_seeds_sc/robustness_seed0.npz \
    --out_dir results_drift_sweep_sc
