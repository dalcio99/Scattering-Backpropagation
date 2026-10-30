"""
gradient_worker_sc.py — single-job worker, SEMICLASSICAL-convention sweep.

Conventions:
  - noise <dW dW^dagger> = dt/2 (symmetric ordering; vacuum <|a|^2> = 1/2);
  - measured record a_out(t) = abar_in + xi(t) + sqrt(kappa) a(t) includes the
    reflected input vacuum (same realisation as the dynamics);
  - all measured quantities (y, err, gradient inputs) are record-based:
    a_hat = (<a_out>_Tm - abar_in)/sqrt(kappa).

The ratio grid: 15 log-spaced values 1e-4 .. 3 (see submit_sc_main.sh) — extended a decade below the sweep-20 grid
because the record-based measurement lowers the SNR failure edge to r ~ 2.5e-3.

Extra tracked quantity: <|a_j|^2>_Tm (total second moment, includes the +1/2
vacuum) alongside |a_hat_j|^2, saved as phot2_* fields.

Self-contained on the cluster: needs only sc_core.py next to it.

Usage
-----
  python gradient_worker_sc.py \\
      --ga0 1.0 --ratio 0.038 --noise_scale 1.0 --seed 0 --mode sigma_y
"""

import os
import sys
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sc_core import (simulate_record, analytic_ss_g0, sb_gradient_matrix,
                     XOR_INPUTS, XOR_TARGETS)

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'results_sc_sweep20')

# ── Physics parameters (identical to sweep 20) ────────────────────────────────
N            = 10
a0_in_sq     = 160.0                              # A, input power per node
amplitude    = float(np.sqrt(a0_in_sq * N))       # = 40.0
out_idx      = 2
KAPPA        = np.ones(N)

# In some cases a target_scale != 1 is needed to rescale your physical output
# so that the target values of your objective function are learnable by the
# physical dynamics. 
target_scale = 1.0 

# ---------------------------------------------------------------------------
# Training hyper-parameters 
# ---------------------------------------------------------------------------
LR                 = 5e-4
NUM_EPOCHS         = 800
T_TRANS            = 40.0
T_SAMPLE_NOISELESS = 20.0
T_SAMPLE_NOISY     = 100.0
DT                 = 0.05
EPS_ERR            = 1e-6
T_SAMPLE_FD        = 400.0
EPS_FD             = 1e-4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(u, v):
    u = np.ravel(u).astype(float)
    v = np.ravel(v).astype(float)
    n1, n2 = np.linalg.norm(u), np.linalg.norm(v)
    if n1 < 1e-14 or n2 < 1e-14:
        return np.nan
    return float(np.clip(np.dot(u, v) / (n1 * n2), -1.0, 1.0))


def _make_J(seed):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((N, N))
    return (R + R.T) / np.sqrt(2 * N)


def _measure(J, g, kappa, a_in, T_sample, dt, noise_scale, rng, a0=None):
    """Simulate and return the SimRecord (record-based estimators inside)."""
    return simulate_record(J, g, kappa, a_in, None, T_TRANS, T_sample, dt,
                           noise_scale, a0=a0, rng=rng)


def _ss_noiseless(J, g, kappa, a_in, T_sample, dt, rng):
    # If the system is linear and noiseless, we return the exact steady state,
    # otherwise we simulate the dynamics
    if g == 0.0: 
        return analytic_ss_g0(J, kappa, a_in)
    return _measure(J, g, kappa, a_in, T_sample, dt, 0.0, rng).a_avg

# ---------------------------------------------------------------------------
# IFT reference gradient (noiseless; physics unchanged vs sweep 20)
# ---------------------------------------------------------------------------

def ift_gradient_batch(J, g, kappa, dt, seed=0):
    rng  = np.random.default_rng(seed)
    grad = np.zeros((N, N))
    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex)
        a_in[:2] = amplitude * inp
        a_ss = _ss_noiseless(J, g, kappa, a_in, T_SAMPLE_NOISELESS, dt, rng)

        # We compute the gradient dC/dJ using the implicit function theorem 
        # We use the formula obtained in the real basis
        u, v  = a_ss.real, a_ss.imag
        a_out = a_in + np.sqrt(kappa) * a_ss
        y     = a_out[out_idx].real
        err   = y - tgt / target_scale

        keff = kappa / 2.0
        A = np.diag(-keff + 2.0 * g * u * v)
        B = J + np.diag(g * (u**2 + 3.0 * v**2))
        C = -J - np.diag(g * (3.0 * u**2 + v**2))
        D = np.diag(-keff - 2.0 * g * u * v)
        Jac = np.block([[A, B], [C, D]])

        e_L = np.zeros(2 * N)
        e_L[out_idx] = 2.0 * err * target_scale**2 * np.sqrt(kappa[out_idx])
        lam   = np.linalg.solve(Jac.T, e_L)
        lam_u, lam_v = lam[:N], lam[N:]

        G = (-(np.outer(lam_u, v) + np.outer(v, lam_u))
              +(np.outer(lam_v, u) + np.outer(u, lam_v)))
        G[np.diag_indices_from(G)] *= 0.5
        grad += G

    return grad / len(XOR_INPUTS)

# ---------------------------------------------------------------------------
# FD-CRN reference gradient (noisy, record-based loss)
# ---------------------------------------------------------------------------

def _mean_loss(J, g, kappa, noise_scale, T_trans, T_sample, dt, seed):
    rng   = np.random.default_rng(seed)          # CRN across +/- eps
    total = 0.0
    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex)
        a_in[:2] = amplitude * inp
        # If the system is linear and noiseless, we return the exact steady state,
        # otherwise we simulate the dynamics. This corresponds to the 'Inference Phase'
        if g == 0.0 and noise_scale == 0.0:
            a_ss = analytic_ss_g0(J, kappa, a_in)
            y = (a_in + np.sqrt(kappa) * a_ss)[out_idx].real
        else:
            a0  = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            rec = simulate_record(J, g, kappa, a_in, None,
                                  T_trans, T_sample, dt, noise_scale,
                                  a0=a0, rng=rng)
            y   = rec.a_out_avg[out_idx].real
        total += (y - tgt / target_scale) ** 2
    return total / len(XOR_INPUTS)


def fd_gradient_crn(J, g, kappa, noise_scale, T_trans, T_sample, dt, eps, seed):
    L_base = _mean_loss(J, g, kappa, noise_scale, T_trans, T_sample, dt, seed)
    grad   = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            J_p = J.copy()
            J_p[i, j] += eps
            if i != j:
                J_p[j, i] += eps
            L_p        = _mean_loss(J_p, g, kappa, noise_scale,
                                    T_trans, T_sample, dt, seed)
            dL         = (L_p - L_base) / eps
            grad[i, j] = dL
            grad[j, i] = dL
    return grad


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
# SB gradient (record-based measurement)
# ---------------------------------------------------------------------------
def sb_gradient_fixed_ratio(J, g, kappa, ratio_target,
                            noise_scale, mode, rng, T_sample, dt):
    dJ_batch  = np.zeros((N, N))
    ratio_sum = 0.0
    n_active  = 0

    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex)
        a_in[:2] = amplitude * inp

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

        rec_p = _measure(J, g, kappa, a_in + da, T_sample, dt, noise_scale,
                         rng, a0=rec.a_end.copy())

        dJ = sb_gradient_matrix(rec.a_hat, rec_p.a_hat - rec.a_hat,
                                beta_eff, mode)
        dJ_batch += dJ

        if np.any(inp > 0.1):
            ratio_sum += abs(da[out_idx]) / amplitude
            n_active  += 1

    ratio_avg = ratio_sum / n_active if n_active else float('nan')
    return dJ_batch / len(XOR_INPUTS), ratio_avg

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fixed_ratio(J0, g, kappa, ratio_target,
                      noise_scale, mode, seed, num_epochs, T_sample, dt):
    rng = np.random.default_rng(seed)
    J   = J0.copy()

    loss_hist       = []
    phot_hist       = []      # coherent |a_hat_j|^2, mean over nodes
    phot_node_hist  = []      # coherent, per node
    phot2_hist      = []      # total <|a_j|^2>_Tm, mean over nodes
    phot2_node_hist = []      # total, per node
    ratio_hist      = []

    for _ep in range(num_epochs):
        dJ, ratio_avg = sb_gradient_fixed_ratio(
            J, g, kappa, ratio_target, noise_scale, mode, rng, T_sample, dt)
        J -= LR * dJ

        el    = 0.0
        ph    = 0.0
        ph2   = 0.0
        ph_n  = np.zeros(N)
        ph2_n = np.zeros(N)
        for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
            a_in = np.zeros(N, dtype=complex)
            a_in[:2] = amplitude * inp
            T_eval = T_sample if noise_scale > 0.0 else T_SAMPLE_NOISELESS
            rec = _measure(J, g, kappa, a_in, T_eval, dt, noise_scale, rng)
            y   = rec.a_out_avg[out_idx].real
            err = y - tgt / target_scale
            el += err**2
            node_phot = np.abs(rec.a_hat)**2
            ph    += float(np.mean(node_phot))
            ph_n  += node_phot
            ph2   += float(np.mean(rec.n2_avg))
            ph2_n += rec.n2_avg

        n = len(XOR_INPUTS)
        loss_hist.append(el / n)
        phot_hist.append(ph / n)
        phot_node_hist.append(ph_n / n)
        phot2_hist.append(ph2 / n)
        phot2_node_hist.append(ph2_n / n)
        ratio_hist.append(ratio_avg)

    return (J,
            np.array(loss_hist),
            np.array(phot_hist),
            np.array(phot_node_hist),
            np.array(phot2_hist),
            np.array(phot2_node_hist),
            np.array(ratio_hist))

# ---------------------------------------------------------------------------
# Single job 
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
    dJ0, _ = sb_gradient_fixed_ratio(
        J0, g, kappa, ratio_target, noise_scale, mode, rng0, T_samp, DT)

    cos_init   = _cosine(dJ0, ref_grad)
    angle_init = float(np.degrees(np.arccos(np.clip(cos_init, -1, 1))))

    t0 = time.time()
    _, lh, ph, ph_node, ph2, ph2_node, rh = train_fixed_ratio(
        J0, g, kappa, ratio_target, noise_scale, mode,
        seed, num_epochs, T_samp, DT)
    elapsed = time.time() - t0

    print(f"  {fname}  angle={angle_init:.1f}°  cos={cos_init:+.3f}"
          f"  ratio_0={rh[0] if len(rh) else float('nan'):.4f}"
          f"  ref={ref_label}  [{elapsed:.0f}s]")

    np.savez(fpath,
             convention='semiclassical_record',
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

# --------------------------------------------------------------------------
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
