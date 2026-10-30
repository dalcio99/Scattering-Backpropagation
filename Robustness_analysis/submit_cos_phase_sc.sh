#!/bin/bash -l

#SBATCH -o ./logs/cosphase.%A_%a.out
#SBATCH -e ./logs/cosphase.%A_%a.err
#SBATCH -D ./
#SBATCH -J cosphasesc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=01:00:00
#SBATCH --mail-type=none
#SBATCH --mail-user=nicola.dalcin@mpl.mpg.de
# #SBATCH --exclude=...   # (site-specific node exclusions removed for release; adapt to your cluster)

# ---------------------------------------------------------------------------
# Gradient degradation under readout phase noise, at INIT: cos(SB, exact) vs
# sigma_phi (clean & noisy dynamics), g=1/320. One job per J0 seed.
# Needs robustness_seed_worker_sweep21.py in this directory (imported for physics).
#
#   mkdir -p logs
#   sbatch --array=0-4 submit_cos_phase.sh
#
# Plot:  conda run -n jax-env python plot_cos_phase.py \
#            --results_dir results_cos_phase_sc --fig_dir Figs/g320
# ---------------------------------------------------------------------------

# Activate your Python environment here, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate sb-env
conda activate jax-env

SEEDS=(0 1 2 3 4)
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID + ${OFFSET:-0} ))]}
echo "cos_phase seed=$SEED"

python cos_phase_worker_sc.py \
    --seed "$SEED" --g 0.003125 --n_phi 20 --n_real 20 \
    --out_dir results_cos_phase_sc
