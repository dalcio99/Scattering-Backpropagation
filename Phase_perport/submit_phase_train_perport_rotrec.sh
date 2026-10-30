#!/bin/bash -l

#SBATCH -o ./logs/phtrainppRR.%A_%a.out
#SBATCH -e ./logs/phtrainppRR.%A_%a.err
#SBATCH -D ./
#SBATCH -J phtrainppRR
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=04:00:00
#SBATCH --mail-type=none
#SBATCH --mail-user=nicola.dalcin@mpl.mpg.de
#SBATCH --exclude=zpg002,zpg009,zp150

# ─────────────────────────────────────────────────────────────────────────────
# Train UNDER INDEPENDENT-PER-PORT readout phase noise, using the PHYSICALLY
# HONEST *rotate-the-full-record* model (--rotate_record): the LO phase error
# rotates the whole time-averaged output record at each port (reflected drive
# included), and the known drive is subtracted afterwards -> carrier leakage
# ~|abar_in| phi_j at the driven ports (0,1). Contrast to
# submit_phase_train_perport.sh (default: rotates the estimator only). g=1/320.
# Grid: 6 sigma_phi × 2 dynamics (clean/noisy) × 2 modes (online/batch) × 5 seeds
#       = 120 jobs.
#
# Cluster deps (flat copy): sc_core.py, robustness_seed_worker_sc.py,
# phase_train_perport_worker.py.
#
#   mkdir -p logs
#   sbatch --array=0-119 submit_phase_train_perport_rotrec.sh
#
# Plot:
#   conda run -n jax-env python ../../../robustness/plot_phase_train.py \
#       --results_dir results_phase_train_perport_rotrec \
#       --fig_dir ../../Figs/Phase_perport_rotrec
# ─────────────────────────────────────────────────────────────────────────────

source ~/scripts/load_conda
conda activate jax-env

SPHI=(0.0 0.02 0.05 0.1 0.2 0.4)
NOISES=(0 1)
BATCHES=(0 1)                 # 0 = online (per-pattern), 1 = batch (mean over dataset)
SEEDS=(0 1 2 3 4)
n_sphi=${#SPHI[@]}; n_noise=${#NOISES[@]}; n_batch=${#BATCHES[@]}; n_seed=${#SEEDS[@]}
total=$(( n_sphi * n_noise * n_batch * n_seed ))

IDX=$(( SLURM_ARRAY_TASK_ID + ${OFFSET:-0} ))
if [ "$IDX" -ge "$total" ]; then echo "IDX $IDX >= $total"; exit 0; fi

sphi_i=$(( IDX / (n_noise * n_batch * n_seed) ))
rem=$(( IDX % (n_noise * n_batch * n_seed) ))
noise_i=$(( rem / (n_batch * n_seed) ))
rem2=$(( rem % (n_batch * n_seed) ))
batch_i=$(( rem2 / n_seed )); seed_i=$(( rem2 % n_seed ))
SP=${SPHI[$sphi_i]}; NS=${NOISES[$noise_i]}; BT=${BATCHES[$batch_i]}; SEED=${SEEDS[$seed_i]}

echo "job $IDX / $total : sigma_phi=$SP noise=$NS batch=$BT seed=$SEED (per-port, rotate_record)"

python phase_train_perport_worker.py \
    --seed "$SEED" --sigma_phi "$SP" --noise "$NS" --batch "$BT" \
    --num_epochs 800 --g 0.003125 --rotate_record \
    --out_dir results_phase_train_perport_rotrec
