#!/bin/bash -l

#SBATCH -o ./logs/robseed21.%A_%a.out
#SBATCH -e ./logs/robseed21.%A_%a.err
#SBATCH -D ./
#SBATCH -J robseedsc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=none
#SBATCH --mail-user=nicola.dalcin@mpl.mpg.de
# #SBATCH --exclude=...   # (site-specific node exclusions removed for release; adapt to your cluster)

# ---------------------------------------------------------------------------
# 5-seed STATIC robustness study, SWEEP-21 conventions (see
# robustness_seed_worker_sc.py docstring):
#   a0_in_sq=160 (amplitude 40), fixed nudge ratio r=0.04, dt=0.05,
#   T_trans=40, T_sample_clean=20 / noisy=200, lr=5e-4, 800 epochs, online σ_y,
#   g = 1/320 hard-coded in the worker (g<|a_in|^2>/kbar^2 = 0.5). Deterministic panels eval at clean window, thermal at noisy.
#
# One job = full 6-panel study for one J0 seed (train clean+noisy, then sweep
# detuning/coupling/loss-rate/nonlinearity/cross-talk/thermal for both networks).
#
# Cost: ~30-35 min/seed local, ~1 h/seed on the cluster. 12 h cap is ample
# headroom (worker writes once at the end → a killed job leaves no output, so a
# clean resubmit is safe). Only robustness_seed_worker_sc.py is needed.
#
#   mkdir -p logs
#   sbatch --array=0-4 submit_robustness_seeds_sc.sh
#
# Aggregate + plot when done (point the plotter at the sweep21 results dir):
#   conda run -n jax-env python plot_robustness_seeds.py \
#       --results_dir results_robustness_seeds_sc
# ---------------------------------------------------------------------------

# Activate your Python environment here, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate sb-env
conda activate jax-env

SEEDS=(0 1 2 3 4)

OFFSET=${OFFSET:-0}
IDX=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
SEED=${SEEDS[$IDX]}

echo "job $IDX : seed=$SEED"

python robustness_seed_worker_sc.py \
    --seed "$SEED" \
    --n_pert 15 \
    --n_sigma 15 \
    --num_epochs 800 \
    --out_dir results_robustness_seeds_sc
