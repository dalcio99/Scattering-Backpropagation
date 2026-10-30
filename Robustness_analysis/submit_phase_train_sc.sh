#!/bin/bash -l

#SBATCH -o ./logs/phtrain.%A_%a.out
#SBATCH -e ./logs/phtrain.%A_%a.err
#SBATCH -D ./
#SBATCH -J phtrainsc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=04:00:00
#SBATCH --mail-type=none
#SBATCH --mail-user=nicola.dalcin@mpl.mpg.de
# #SBATCH --exclude=...   # (site-specific node exclusions removed for release; adapt to your cluster)

# ---------------------------------------------------------------------------
# Train UNDER readout phase noise (full homodyne, present during training),
# g=1/320. Does SB still converge when the per-step gradient is only weakly
# aligned — and does BATCHING over the XOR dataset average the phase noise out?
# Grid: 6 sigma_phi × 2 dynamics (clean/noisy) × 2 modes (online/batch) × 5 seeds
#       = 120 jobs.
# Needs robustness_seed_worker_sweep21.py in this directory (imported for physics).
#
#   mkdir -p logs
#   sbatch --array=0-119 submit_phase_train.sh
#
# Plot:  conda run -n jax-env python plot_phase_train.py \
#            --results_dir results_phase_train_sc --fig_dir Figs/g320
# ---------------------------------------------------------------------------

# Activate your Python environment here, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate sb-env
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

echo "job $IDX / $total : sigma_phi=$SP noise=$NS batch=$BT seed=$SEED"

python phase_train_worker_sc.py \
    --seed "$SEED" --sigma_phi "$SP" --noise "$NS" --batch "$BT" \
    --num_epochs 800 --g 0.003125 \
    --out_dir results_phase_train_sc
