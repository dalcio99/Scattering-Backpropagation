#!/bin/bash -l

#SBATCH -o ./logs/cosphaseppRR.%A_%a.out
#SBATCH -e ./logs/cosphaseppRR.%A_%a.err
#SBATCH -D ./
#SBATCH -J cosphaseppRR
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=01:00:00
#SBATCH --mail-type=none
#SBATCH --mail-user=nicola.dalcin@mpl.mpg.de
#SBATCH --exclude=zpg002,zpg009,zp150

# ─────────────────────────────────────────────────────────────────────────────
# Gradient degradation under INDEPENDENT-PER-PORT readout phase noise, at INIT,
# using the PHYSICALLY HONEST *rotate-the-full-record* model (--rotate_record):
#   measured record  <a_out,j>^meas = e^{i phi_j} <a_out,j>_Tm  (drive included),
#   estimator        a_hat_j = (<a_out,j>^meas - abar_in,j)/sqrt(kappa),
# i.e. the LO phase error rotates the whole detected field and the KNOWN drive is
# subtracted afterwards -> carrier leakage ~|abar_in| phi at the driven ports.
# Contrast to submit_cos_phase_perport.sh (default: phase rotates the estimator
# only / phase-clean drive subtraction). g=1/320, one job per J0 seed.
#
# The FD-CRN noisy reference is PHASE-FREE, hence identical to the default run.
# To skip recomputation (~1 min/seed), copy the cached references first:
#   mkdir -p results_cos_phase_perport_rotrec
#   cp results_cos_phase_perport/fd_grad_*.npy results_cos_phase_perport_rotrec/
#
# Cluster deps (flat copy): sc_core.py, robustness_seed_worker_sc.py,
# gradient_worker_sc.py, cos_phase_perport_worker.py.
#
#   mkdir -p logs
#   sbatch --array=0-4 submit_cos_phase_perport_rotrec.sh
#
# Plot:
#   conda run -n jax-env python ../../../robustness/plot_cos_phase.py \
#       --results_dir results_cos_phase_perport_rotrec \
#       --fig_dir ../../Figs/Phase_perport_rotrec
# ─────────────────────────────────────────────────────────────────────────────

source ~/scripts/load_conda
conda activate jax-env

SEEDS=(0 1 2 3 4)
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID + ${OFFSET:-0} ))]}
echo "cos_phase (per-port, rotate_record) seed=$SEED"

python cos_phase_perport_worker.py \
    --seed "$SEED" --g 0.003125 --n_phi 20 --n_real 20 --rotate_record \
    --out_dir results_cos_phase_perport_rotrec
