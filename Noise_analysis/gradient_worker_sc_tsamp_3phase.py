"""
gradient_worker_sc_tsamp_3phase.py — T_m study, 3-phase SB, semiclassical sweep.

Fixed point: ga0_target = 1.0, ratio = 0.05, noise_scale = 1.0;
T_m ∈ {25, 50, 100, 200, 400}; 500 epochs, lr = 5e-4. Loss evaluated at
<y>_{T_m} (record-based). File naming identical to the 2-phase tsamp study
(plot_sweep20_tsamp.py works via --results_dir); outputs go to a separate
directory so 2-phase and 3-phase results never mix.

Usage
-----
  python gradient_worker_sc_tsamp_3phase.py --t_sample 100 --seed 0 --mode sigma_y
"""

import os
import sys
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradient_worker_sc import (_make_J, N, a0_in_sq, amplitude, KAPPA,
                                T_TRANS, T_SAMPLE_NOISELESS, DT, LR, EPS_ERR)
from gradient_worker_sc_3phase import train_fixed_ratio_3phase

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'results_tsamp_sc_3phase')

DEFAULT_GA0   = 1.0
DEFAULT_RATIO = 0.05
NOISE_SCALE   = 1.0
NUM_EPOCHS    = 500
T_SAMP_VALS   = [25, 50, 100, 200, 400]


def run_one(ga0_target, ratio_target, t_sample, seed, mode, num_epochs):
    tag   = (f"{mode}_ga0{ga0_target:.3g}_ratio{ratio_target:.6g}"
             f"_noise{NOISE_SCALE}_tsamp{t_sample}_seed{seed}")
    fname = tag + '.npz'
    fpath = os.path.join(OUT_DIR, fname)

    if os.path.exists(fpath):
        print(f"  skip (exists): {fname}")
        return

    g     = ga0_target / a0_in_sq if ga0_target > 0.0 else 0.0
    kappa = KAPPA.copy()
    J0    = _make_J(seed)

    t0 = time.time()
    _, lh, ph, ph_node, ph2, ph2_node, rh = train_fixed_ratio_3phase(
        J0, g, kappa, ratio_target, NOISE_SCALE, mode,
        seed, num_epochs, t_sample, DT)
    elapsed = time.time() - t0

    print(f"  {fname}  loss_last25={lh[-25:].mean():.3e}"
          f"  phot_final={float(ph[-1]):.2f}  [{elapsed:.0f}s]")

    np.savez(fpath,
             convention='semiclassical_record',
             method='3phase',
             mode=mode,
             ga0_target=ga0_target,
             g=g,
             a0_in_sq=a0_in_sq,
             N=N,
             kappa_val=1.0,
             amplitude=amplitude,
             ratio_target=ratio_target,
             noise_scale=NOISE_SCALE,
             t_sample=t_sample,
             seed=seed,
             T_trans=T_TRANS,
             T_sample_noiseless=T_SAMPLE_NOISELESS,
             dt=DT,
             lr=LR,
             num_epochs=num_epochs,
             eps_err=EPS_ERR,
             loss_history=lh,
             phot_history=ph,
             phot_node_history=ph_node,
             phot_final=float(ph[-1]),
             phot_node_final=ph_node[-1],
             phot2_history=ph2,
             phot2_node_history=ph2_node,
             phot2_node_final=ph2_node[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ga0',        type=float, default=DEFAULT_GA0)
    p.add_argument('--ratio',      type=float, default=DEFAULT_RATIO)
    p.add_argument('--t_sample',   type=float, required=True)
    p.add_argument('--seed',       type=int,   default=0)
    p.add_argument('--mode',       type=str,   default='sigma_y',
                   choices=['sigma_x', 'sigma_y'])
    p.add_argument('--num_epochs', type=int,   default=NUM_EPOCHS)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    run_one(args.ga0, args.ratio, args.t_sample, args.seed,
            args.mode, args.num_epochs)


if __name__ == '__main__':
    main()
