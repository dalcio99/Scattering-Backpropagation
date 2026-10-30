#!/bin/bash -l

#SBATCH -o ./logs/gradsc3.%A_%a.out
#SBATCH -e ./logs/gradsc3.%A_%a.err
#SBATCH -D ./
#SBATCH -J gradsc3
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

# Semiclassical-convention sweep, 3-PHASE (symmetric-nudging) SB:
#   noise <dW dW^dagger> = dt/2, record-based estimators (reflected vacuum in).
# Fixed: a0_in_sq=160, lr=5e-4, T_sample_noisy=100, dt=0.05, T_trans=40, 800 ep.
# Files needed on the cluster: sc_core.py, gradient_worker_sc.py,
# gradient_worker_sc_3phase.py, this script.
#
# Ratio grid: 15 log-spaced values 1e-4 .. 3 (step x2.0883). The record-based
# measurement lowers the SNR failure edge to r ~ 2.5e-3 (T_m = 100), so the
# grid extends a decade below it: expect ~4-5 SNR-failing points, ~5 plateau,
# ~5 Taylor-limited.
#
# Grid: 15 ratio × 9 ga0 × 2 noise × 4 seeds × 2 modes = 2160 jobs
# Submit in 3 batches (cluster MaxArraySize ≤ 1000):
#
#   mkdir -p logs results_sc_sweep22_3phase
#   sbatch --array=0-719                           submit_sc_3phase_main.sh
#   sbatch --array=0-719 --export=ALL,OFFSET=720   submit_sc_3phase_main.sh
#   sbatch --array=0-719 --export=ALL,OFFSET=1440  submit_sc_3phase_main.sh
#
# Additional seeds (e.g. 4-7, for 8-seed statistics as in sweep 21): set
# SEED_OFFSET, which shifts the SEEDS list without touching the grid decode:
#
#   sbatch --array=0-719 --export=ALL,SEED_OFFSET=4               submit_sc_3phase_main.sh
#   sbatch --array=0-719 --export=ALL,OFFSET=720,SEED_OFFSET=4    submit_sc_3phase_main.sh
#   sbatch --array=0-719 --export=ALL,OFFSET=1440,SEED_OFFSET=4   submit_sc_3phase_main.sh
#
# Workers skip existing output files, and FD-CRN caches are per-seed, so
# re-submissions and seed extensions never clash with completed runs.
#
# FD-CRN cache files are recomputed on demand by the workers (record-based
# loss => the old sweep-20 FD caches do NOT apply; do not copy them).

# Activate your Python environment here, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate sb-env
conda activate jax-env

# ── Parameter grid (must match gradient_worker_sc_3phase.py) ────────────────────────
RATIO_VALUES=(0.0001 0.000208831 0.000436103 0.000910716 0.00190186 0.00397166 0.00829404 0.0173205 0.0361705 0.0755352 0.157741 0.329411 0.687911 1.43657 3.0)
GA0_VALUES=(0.0 0.1 0.3 0.5 1.0 2.0 3.0 5.0 8.0)
NOISE_VALUES=(0.0 1.0)
SEEDS=(0 1 2 3)
MODES=(sigma_x sigma_y)

n_ratio=${#RATIO_VALUES[@]}
n_ga0=${#GA0_VALUES[@]}
n_noise=${#NOISE_VALUES[@]}
n_seed=${#SEEDS[@]}
n_mode=${#MODES[@]}

total_jobs=$(( n_ratio * n_ga0 * n_noise * n_seed * n_mode ))

OFFSET=${OFFSET:-0}
IDX=$(( SLURM_ARRAY_TASK_ID + OFFSET ))

if [[ $IDX -ge $total_jobs ]]; then
    echo "Error: IDX=$IDX >= total_jobs=$total_jobs"
    exit 1
fi

# ── Decode flat index → (ratio, ga0, noise, seed, mode) ──────────────────────
mode_idx=$(( IDX % n_mode  ));  IDX=$(( IDX / n_mode  ))
seed_idx=$(( IDX % n_seed  ));  IDX=$(( IDX / n_seed  ))
noise_idx=$(( IDX % n_noise )); IDX=$(( IDX / n_noise ))
ga0_idx=$(( IDX % n_ga0   ));   IDX=$(( IDX / n_ga0   ))
ratio_idx=$(( IDX % n_ratio ))

RATIO=${RATIO_VALUES[$ratio_idx]}
GA0=${GA0_VALUES[$ga0_idx]}
NOISE=${NOISE_VALUES[$noise_idx]}
SEED=$(( ${SEEDS[$seed_idx]} + ${SEED_OFFSET:-0} ))
MODE=${MODES[$mode_idx]}

echo "Job $((SLURM_ARRAY_TASK_ID + ${OFFSET:-0})) / $total_jobs: mode=$MODE ga0=$GA0 ratio=$RATIO noise=$NOISE seed=$SEED"

srun python gradient_worker_sc_3phase.py \
    --ga0         $GA0   \
    --ratio       $RATIO \
    --noise_scale $NOISE \
    --seed        $SEED  \
    --mode        $MODE
