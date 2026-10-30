"""
gradient_worker_sc_3phase.py — single-job worker, semiclassical sweep, 3-phase SB.

Three-phase (symmetric-nudging) analog of gradient_worker_sc.py with the same
sc_core.py physics (noise <dW dW^dagger> = dt/2, record-based estimators
including the reflected input vacuum). 

Gradient signal: central difference of the two nudged records,
    Delta a_hat = a_hat(+da) - a_hat(-da),
which cancels the even-order nonlinear corrections: O(beta^2) systematic error
instead of O(beta) for the 2-phase forward difference (SI, symmetric nudging).

  sigma_y (da[out] = -i beta err): dJ = -(1/beta) Re[sym outer(a_hat, Delta)]
  sigma_x (da[out] = +  beta err): dJ = -(1/beta) Im[sym outer(a_hat, Delta)]
  (factor 1/beta, not 2/beta: Delta ~ 2 * one-sided response; diagonal halved)

Everything else (grid, J init, beta_eff = ratio*amplitude/|err|, references,
file naming) matches gradient_worker_sc.py, so plot_sweep.py works on the
output directory unchanged. Cost: 3 simulations per (pattern, epoch) vs 2.

Self-contained on the cluster with sc_core.py + gradient_worker_sc.py.

Usage
-----
  python gradient_worker_sc_3phase.py \\
      --ga0 1.0 --ratio 0.038 --noise_scale 1.0 --seed 0 --mode sigma_y
"""

import os
import sys
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradient_worker_sc import (
    _cosine, _make_J, _measure, ift_gradient_batch, fd_gradient_crn,
    N, a0_in_sq, amplitude, out_idx, KAPPA, target_scale,
    LR, NUM_EPOCHS, T_TRANS, T_SAMPLE_NOISELESS, T_SAMPLE_NOISY, DT,
    EPS_ERR, T_SAMPLE_FD, EPS_FD,
)
from sc_core import XOR_INPUTS, XOR_TARGETS

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'results_sc_sweep22_3phase')

# ---------------------------------------------------------------------------
# 3-phase gradient formula (record-based inputs)
# ---------------------------------------------------------------------------

def sb_gradient_matrix_3phase(a_hat, delta_a_hat_cd, beta, mode='sigma_y'):
    """dJ from the free record estimate a_hat and the central-difference
    numerator delta_a_hat_cd = a_hat_plus - a_hat_minus (NOT divided by 2beta).
    J -= lr * dJ descends. Exact at g = 0 for any beta."""
    M_cplx = np.outer(a_hat, delta_a_hat_cd) + np.outer(delta_a_hat_cd, a_hat)
    M = M_cplx.real if mode == 'sigma_y' else M_cplx.imag
    M[np.diag_indices_from(M)] *= 0.5
    return -(1.0 / beta) * M


def sb_gradient_fixed_ratio_3phase(J, g, kappa, ratio_target,
                                   noise_scale, mode, rng, T_sample, dt):
    dJ_batch  = np.zeros((N, N))
    ratio_sum = 0.0
    n_active  = 0

    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex)
        a_in[:2] = amplitude * inp

        # Phase 1 — free record
        rec = _measure(J, g, kappa, a_in, T_sample, dt, noise_scale, rng)
        y   = rec.a_out_avg[out_idx].real
        err = y - tgt / target_scale

        abs_err  = max(abs(err), EPS_ERR)
        beta_eff = ratio_target * amplitude / abs_err

        da = np.zeros(N, dtype=complex)
        if mode == 'sigma_y':
            da[out_idx] = -1j * beta_eff * err
        else:
            da[out_idx] = beta_eff * err

        # Phase 2 — positive perturbation (warm start from free steady state)
        rec_p = _measure(J, g, kappa, a_in + da, T_sample, dt, noise_scale,
                         rng, a0=rec.a_end.copy())
        # Phase 3 — negative perturbation
        rec_m = _measure(J, g, kappa, a_in - da, T_sample, dt, noise_scale,
                         rng, a0=rec.a_end.copy())

        delta_cd = rec_p.a_hat - rec_m.a_hat
        if not np.all(np.isfinite(delta_cd)):
            continue

        dJ_batch += sb_gradient_matrix_3phase(rec.a_hat, delta_cd,
                                              beta_eff, mode)

        if np.any(inp > 0.1):
            ratio_sum += abs(da[out_idx]) / amplitude
            n_active  += 1

    ratio_avg = ratio_sum / n_active if n_active else float('nan')
    return dJ_batch / len(XOR_INPUTS), ratio_avg

# ---------------------------------------------------------------------------
# Training loop (same evaluation as gradient_worker_sc)
# ---------------------------------------------------------------------------

def train_fixed_ratio_3phase(J0, g, kappa, ratio_target,
                             noise_scale, mode, seed, num_epochs,
                             T_sample, dt):
    rng = np.random.default_rng(seed)
    J   = J0.copy()

    loss_hist, phot_hist, phot_node_hist = [], [], []
    phot2_hist, phot2_node_hist, ratio_hist = [], [], []

    for _ep in range(num_epochs):
        dJ, ratio_avg = sb_gradient_fixed_ratio_3phase(
            J, g, kappa, ratio_target, noise_scale, mode, rng, T_sample, dt)
        J -= LR * dJ

        el, ph, ph2 = 0.0, 0.0, 0.0
        ph_n, ph2_n = np.zeros(N), np.zeros(N)
        for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
            a_in = np.zeros(N, dtype=complex)
            a_in[:2] = amplitude * inp
            T_eval = T_sample if noise_scale > 0.0 else T_SAMPLE_NOISELESS
            rec = _measure(J, g, kappa, a_in, T_eval, dt, noise_scale, rng)
            y   = rec.a_out_avg[out_idx].real
            el += (y - tgt / target_scale) ** 2
            node_phot = np.abs(rec.a_hat) ** 2
            ph    += float(np.mean(node_phot));  ph_n  += node_phot
            ph2   += float(np.mean(rec.n2_avg)); ph2_n += rec.n2_avg

        n = len(XOR_INPUTS)
        loss_hist.append(el / n)
        phot_hist.append(ph / n)
        phot_node_hist.append(ph_n / n)
        phot2_hist.append(ph2 / n)
        phot2_node_hist.append(ph2_n / n)
        ratio_hist.append(ratio_avg)

    return (J, np.array(loss_hist), np.array(phot_hist),
            np.array(phot_node_hist), np.array(phot2_hist),
            np.array(phot2_node_hist), np.array(ratio_hist))

# --------------------------------------------------------------------------
# FD-CRN cache local to this sweep's output directory 
# --------------------------------------------------------------------------

def _fd_grad_cached(J, g, kappa, noise_scale, seed):
    tag   = f"fd_grad_g{g:.6g}_noise{noise_scale}_seed{seed}.npy"
    fpath = os.path.join(OUT_DIR, tag)
    if os.path.exists(fpath):
        return np.load(fpath)
    grad = fd_gradient_crn(J, g, kappa, noise_scale,
                           T_TRANS, T_SAMPLE_FD, DT, EPS_FD, seed)
    tmp = fpath + f'.tmp{os.getpid()}'
    np.save(tmp, grad)
    os.replace(tmp + '.npy', fpath)
    return grad

# ---------------------------------------------------------------------------
# Single Job
# ---------------------------------------------------------------------------
def run_one(ga0_target, ratio_target, noise_scale, seed, mode, num_epochs):
    tag   = f"{mode}_ga0{ga0_target:.3g}_ratio{ratio_target:.6g}"
    fname = f"{tag}_noise{noise_scale}_seed{seed}.npz"
    fpath = os.path.join(OUT_DIR, fname)

    if os.path.exists(fpath):
        print(f"  skip (exists): {fname}")
        return

    g      = ga0_target / a0_in_sq if ga0_target > 0.0 else 0.0
    kappa  = KAPPA.copy()
    T_samp = T_SAMPLE_NOISY if noise_scale > 0 else T_SAMPLE_NOISELESS
    J0     = _make_J(seed)

    if noise_scale == 0.0:
        ref_grad  = ift_gradient_batch(J0, g, kappa, DT, seed=seed)
        ref_label = 'IFT'
    else:
        ref_grad  = _fd_grad_cached(J0, g, kappa, noise_scale, seed)
        ref_label = 'FD-CRN'

    rng0 = np.random.default_rng(seed)
    dJ0, _ = sb_gradient_fixed_ratio_3phase(
        J0, g, kappa, ratio_target, noise_scale, mode, rng0, T_samp, DT)

    cos_init   = _cosine(dJ0, ref_grad)
    angle_init = float(np.degrees(np.arccos(np.clip(cos_init, -1, 1))))

    t0 = time.time()
    _, lh, ph, ph_node, ph2, ph2_node, rh = train_fixed_ratio_3phase(
        J0, g, kappa, ratio_target, noise_scale, mode,
        seed, num_epochs, T_samp, DT)
    elapsed = time.time() - t0

    print(f"  {fname}  angle={angle_init:.1f}°  cos={cos_init:+.3f}"
          f"  ratio_0={rh[0] if len(rh) else float('nan'):.4f}"
          f"  ref={ref_label}  [{elapsed:.0f}s]")

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
             noise_scale=noise_scale,
             seed=seed,
             T_trans=T_TRANS,
             T_sample=T_samp,
             T_sample_fd=T_SAMPLE_FD,
             dt=DT,
             lr=LR,
             num_epochs=num_epochs,
             eps_err=EPS_ERR,
             eps_fd=EPS_FD,
             ref_label=ref_label,
             loss_history=lh,
             phot_history=ph,
             phot_node_history=ph_node,
             phot_node_final=ph_node[-1],
             phot2_history=ph2,
             phot2_node_history=ph2_node,
             phot2_node_final=ph2_node[-1],
             ratio_history=rh,
             angle_deg_init=angle_init,
             cos_init=cos_init,
             ratio_init=ratio_target)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ga0',         type=float, required=True)
    p.add_argument('--ratio',       type=float, required=True)
    p.add_argument('--noise_scale', type=float, default=0.0)
    p.add_argument('--seed',        type=int,   default=0)
    p.add_argument('--mode',        type=str,   default='sigma_y',
                   choices=['sigma_x', 'sigma_y'])
    p.add_argument('--num_epochs',  type=int,   default=NUM_EPOCHS)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    run_one(args.ga0, args.ratio, args.noise_scale, args.seed,
            args.mode, args.num_epochs)


if __name__ == '__main__':
    main()
