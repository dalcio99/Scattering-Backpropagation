"""
gradient_worker_nl_sc.py — "other nonlinearities" study, SEMICLASSICAL shot 
noise convention.

Port of Corrected_XOR/Other_nonlinearities/gradient_worker_nl.py to the
sc_core.py physics: noise <dW dW^dagger> = dt/2 (truncated Wigner) and
record-based estimators (reflected input vacuum in the measured record,
a_hat = (<a_out>_Tm - abar_in)/sqrt(kappa)).

Everything else matches the old study exactly: N = 10 fully connected,
a0_in_sq = 160, out_idx = 2, fixed nudge ratio 0.04, lr = 5e-4, 800 epochs,
T_m = 20 (clean) / 200 (noisy), dt = 0.05, sigma_y; nonlinearities (n = |a|^2):

  'kerr'      NL = -i g n a
  'saturable' NL = -i g n/(1+n/n_s) a          (n_s = 50)
  'tpa'       NL = -i g n a - 1/2 gamma n a    (gamma = 0.3 g)

The SB gradient uses ONLY the measured records and is identical across
nonlinearities; only the drift callable and the IFT validation Jacobian know
the nonlinearity.

Cluster files: sc_core.py (parent dir or alongside), this worker, submit
script. File naming identical to the old study:
  {nl}_{mode}_ga0{ga0:.3g}_noise{noise}_seed{seed}.npz

Usage
-----
  python gradient_worker_nl_sc.py --nl saturable --ga0 3.0 --noise_scale 1.0 \\
      --seed 0 --mode sigma_y
"""

import os
import sys
import time
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE),
          os.path.join(os.path.dirname(HERE), 'Noise_analysis')):  # shared modules (sc_core) live in Code/Noise_analysis
    if p not in sys.path:
        sys.path.insert(0, p)
from sc_core import (simulate_record, analytic_ss_g0, sb_gradient_matrix,
                     XOR_INPUTS, XOR_TARGETS)

OUT_DIR = os.path.join(HERE, 'results_nl_sc')

# Physics parameters
N            = 10
a0_in_sq     = 160.0
amplitude    = float(np.sqrt(a0_in_sq * N))   # 40
out_idx      = 2
KAPPA        = np.ones(N)
target_scale = 1.0

LR                 = 5e-4
NUM_EPOCHS         = 800
T_TRANS            = 40.0
T_SAMPLE_NOISELESS = 20.0
T_SAMPLE_NOISY     = 200.0
DT                 = 0.05
EPS_ERR            = 1e-6
T_SAMPLE_FD        = 400.0
EPS_FD             = 1e-4

RATIO_FIXED = 0.04

NL_DEFAULTS = {
    'kerr':      {},
    'saturable': {'n_s': 50.0},
    'tpa':       {'tpa_ratio': 0.3},
}


# Nonlinearity callables for sc_core.drift (batch-safe, elementwise)

def _nl_params(nl, g, params):
    params = params or {}
    if nl == 'kerr':
        return np.inf, 0.0
    if nl == 'saturable':
        return float(params.get('n_s', 200.0)), 0.0
    if nl == 'tpa':
        return np.inf, float(params.get('tpa_ratio', 1.0)) * g
    raise ValueError(f"unknown nonlinearity '{nl}'")


def phi_dphi(n, n_s):
    """Reactive shape phi(n) and derivative (Kerr if n_s = inf)."""
    if np.isinf(n_s):
        return n, np.ones_like(n)
    d = 1.0 + n / n_s
    return n / d, 1.0 / d**2


def make_nl(nl, g, params):
    """Return the callable nl_fn(a, g) for sc_core.simulate_record."""
    n_s, gamma = _nl_params(nl, g, params)

    def nl_fn(a, g_):
        n = np.abs(a) ** 2
        phi, _ = phi_dphi(n, n_s)
        return (-1j * g_ * phi - 0.5 * gamma * n) * a

    return nl_fn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cosine(u, v):
    u = np.ravel(u).astype(float); v = np.ravel(v).astype(float)
    n1, n2 = np.linalg.norm(u), np.linalg.norm(v)
    if n1 < 1e-14 or n2 < 1e-14:
        return np.nan
    return float(np.clip(np.dot(u, v) / (n1 * n2), -1.0, 1.0))


def make_J(seed):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((N, N))
    return (R + R.T) / np.sqrt(2 * N)


def _measure(J, g, kappa, a_in, nl_fn, T_sample, dt, noise_scale, rng, a0=None):
    return simulate_record(J, g, kappa, a_in, None, T_TRANS, T_sample, dt,
                           noise_scale, a0=a0, rng=rng, nl=nl_fn)


def _ss_clean(J, g, kappa, a_in, nl_fn, dt, rng):
    if g == 0.0:
        return analytic_ss_g0(J, kappa, a_in)
    return _measure(J, g, kappa, a_in, nl_fn, T_SAMPLE_NOISELESS, dt, 0.0,
                    rng).a_avg

# ---------------------------------------------------------------------------
# IFT reference (noiseless; convention-independent, from nl_core.py)
# ---------------------------------------------------------------------------
def ift_jacobian(J, g, kappa, u, v, nl, params):
    n = u**2 + v**2
    n_s, gamma = _nl_params(nl, g, params)
    phi, dphi = phi_dphi(n, n_s)
    keff = kappa / 2.0
    A = np.diag(-keff + 2.0*g*u*v*dphi        - 0.5*gamma*(n + 2.0*u**2))
    B =  J + np.diag(g*(phi + 2.0*v**2*dphi)   - 0.5*gamma*(2.0*u*v))
    C = -J + np.diag(-g*(phi + 2.0*u**2*dphi)  - 0.5*gamma*(2.0*u*v))
    D = np.diag(-keff - 2.0*g*u*v*dphi         - 0.5*gamma*(n + 2.0*v**2))
    return np.block([[A, B], [C, D]])


def ift_gradient_batch(J, g, kappa, nl, params, seed=0):
    rng = np.random.default_rng(seed)
    nl_fn = make_nl(nl, g, params)
    grad = np.zeros((N, N))
    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex); a_in[:2] = amplitude * inp
        a_ss = _ss_clean(J, g, kappa, a_in, nl_fn, DT, rng)
        u, v = a_ss.real, a_ss.imag
        y = (a_in + np.sqrt(kappa) * a_ss)[out_idx].real
        err = y - tgt / target_scale
        Jac = ift_jacobian(J, g, kappa, u, v, nl, params)
        e_L = np.zeros(2 * N)
        e_L[out_idx] = 2.0 * err * target_scale**2 * np.sqrt(kappa[out_idx])
        lam = np.linalg.solve(Jac.T, e_L)
        lam_u, lam_v = lam[:N], lam[N:]
        G = (-(np.outer(lam_u, v) + np.outer(v, lam_u))
             + (np.outer(lam_v, u) + np.outer(u, lam_v)))
        G[np.diag_indices_from(G)] *= 0.5
        grad += G
    return grad / len(XOR_INPUTS)

# ---------------------------------------------------------------------------
# FD-CRN reference (noisy, record-based loss)
# ---------------------------------------------------------------------------
def _mean_loss(J, g, kappa, nl_fn, noise_scale, T_sample, seed):
    rng = np.random.default_rng(seed)          # CRN across +/- eps
    total = 0.0
    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex); a_in[:2] = amplitude * inp
        if g == 0.0 and noise_scale == 0.0:
            a_ss = analytic_ss_g0(J, kappa, a_in)
            y = (a_in + np.sqrt(kappa) * a_ss)[out_idx].real
        else:
            a0 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            rec = _measure(J, g, kappa, a_in, nl_fn, T_sample, DT,
                           noise_scale, rng, a0=a0)
            y = rec.a_out_avg[out_idx].real
        total += (y - tgt / target_scale) ** 2
    return total / len(XOR_INPUTS)


def _fd_grad_cached(J, g, kappa, nl, params, noise_scale, seed):
    tag = f"fd_grad_{nl}_g{g:.6g}_noise{noise_scale}_seed{seed}.npy"
    fpath = os.path.join(OUT_DIR, tag)
    if os.path.exists(fpath):
        return np.load(fpath)
    nl_fn = make_nl(nl, g, params)
    L0 = _mean_loss(J, g, kappa, nl_fn, noise_scale, T_SAMPLE_FD, seed)
    grad = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            Jp = J.copy()
            Jp[i, j] += EPS_FD
            if i != j:
                Jp[j, i] += EPS_FD
            Lp = _mean_loss(Jp, g, kappa, nl_fn, noise_scale, T_SAMPLE_FD, seed)
            grad[i, j] = grad[j, i] = (Lp - L0) / EPS_FD
    tmp = fpath + f'.tmp{os.getpid()}'
    np.save(tmp, grad)
    os.replace(tmp + '.npy', fpath)
    return grad

# ---------------------------------------------------------------------------
# SB gradient (record-based, fixed-ratio)
# ---------------------------------------------------------------------------
def sb_gradient_fixed_ratio(J, g, kappa, nl_fn, ratio,
                            noise_scale, mode, rng, T_sample, dt):
    dJ = np.zeros((N, N))
    ratio_sum = 0.0; n_active = 0
    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex); a_in[:2] = amplitude * inp
        rec = _measure(J, g, kappa, a_in, nl_fn, T_sample, dt, noise_scale, rng)
        err = rec.a_out_avg[out_idx].real - tgt / target_scale
        beta = ratio * amplitude / max(abs(err), EPS_ERR)
        da = np.zeros(N, dtype=complex)
        da[out_idx] = (-1j if mode == 'sigma_y' else 1.0) * beta * err
        rec_p = _measure(J, g, kappa, a_in + da, nl_fn, T_sample, dt,
                         noise_scale, rng, a0=rec.a_end.copy())
        dJ += sb_gradient_matrix(rec.a_hat, rec_p.a_hat - rec.a_hat, beta, mode)
        if np.any(inp > 0.1):
            ratio_sum += abs(da[out_idx]) / amplitude; n_active += 1
    ratio_avg = ratio_sum / n_active if n_active else float('nan')
    return dJ / len(XOR_INPUTS), ratio_avg


def train_fixed_ratio(J0, g, kappa, nl_fn, ratio,
                      noise_scale, mode, seed, num_epochs, T_sample, dt):
    rng = np.random.default_rng(seed)
    J = J0.copy()
    loss_hist, phot_hist, phot2_hist, ratio_hist = [], [], [], []
    for _ep in range(num_epochs):
        dJ, ratio_avg = sb_gradient_fixed_ratio(
            J, g, kappa, nl_fn, ratio, noise_scale, mode, rng, T_sample, dt)
        J -= LR * dJ
        el = ph = ph2 = 0.0
        for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
            a_in = np.zeros(N, dtype=complex); a_in[:2] = amplitude * inp
            T_eval = T_sample if noise_scale > 0.0 else T_SAMPLE_NOISELESS
            rec = _measure(J, g, kappa, a_in, nl_fn, T_eval, dt,
                           noise_scale, rng)
            el  += (rec.a_out_avg[out_idx].real - tgt / target_scale) ** 2
            ph  += float(np.mean(np.abs(rec.a_hat) ** 2))
            ph2 += float(np.mean(rec.n2_avg))
        n = len(XOR_INPUTS)
        loss_hist.append(el / n); phot_hist.append(ph / n)
        phot2_hist.append(ph2 / n); ratio_hist.append(ratio_avg)
    return (J, np.array(loss_hist), np.array(phot_hist),
            np.array(phot2_hist), np.array(ratio_hist))

# ---------------------------------------------------------------------------
#  Single job
# ---------------------------------------------------------------------------
def run_one(nl, ga0, noise_scale, seed, mode, params, num_epochs):
    tag = f"{nl}_{mode}_ga0{ga0:.3g}_noise{noise_scale}_seed{seed}"
    fpath = os.path.join(OUT_DIR, tag + '.npz')
    if os.path.exists(fpath):
        print(f"  skip (exists): {tag}"); return

    g = ga0 / a0_in_sq if ga0 > 0.0 else 0.0
    kappa = KAPPA.copy()
    T_samp = T_SAMPLE_NOISY if noise_scale > 0 else T_SAMPLE_NOISELESS
    J0 = make_J(seed)
    nl_fn = make_nl(nl, g, params)

    if noise_scale == 0.0:
        ref = ift_gradient_batch(J0, g, kappa, nl, params, seed=seed)
        ref_label = 'IFT'
    else:
        ref = _fd_grad_cached(J0, g, kappa, nl, params, noise_scale, seed)
        ref_label = 'FD-CRN'

    dJ0, _ = sb_gradient_fixed_ratio(J0, g, kappa, nl_fn, RATIO_FIXED,
                                     noise_scale, mode,
                                     np.random.default_rng(seed), T_samp, DT)
    cos_init = _cosine(dJ0, ref)
    angle_init = float(np.degrees(np.arccos(np.clip(cos_init, -1, 1))))

    t0 = time.time()
    _, lh, ph, ph2, rh = train_fixed_ratio(J0, g, kappa, nl_fn, RATIO_FIXED,
                                           noise_scale, mode, seed,
                                           num_epochs, T_samp, DT)
    elapsed = time.time() - t0

    print(f"  {tag}  L0={lh[0]:.2e} Lmin={lh.min():.2e} Lend={lh[-1]:.2e}  "
          f"angle={angle_init:.1f}° cos={cos_init:+.3f}  ref={ref_label} [{elapsed:.0f}s]")

    np.savez(fpath,
             convention='semiclassical_record',
             nl=nl, mode=mode, ga0=ga0, g=g, nl_params=json.dumps(params),
             a0_in_sq=a0_in_sq, N=N, kappa_val=1.0, amplitude=amplitude,
             ratio_fixed=RATIO_FIXED, noise_scale=noise_scale, seed=seed,
             T_trans=T_TRANS, T_sample=T_samp, T_sample_fd=T_SAMPLE_FD,
             dt=DT, lr=LR, num_epochs=num_epochs,
             eps_err=EPS_ERR, eps_fd=EPS_FD, ref_label=ref_label,
             loss_history=lh, phot_history=ph, phot2_history=ph2,
             ratio_history=rh,
             angle_deg_init=angle_init, cos_init=cos_init)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nl',          type=str,   required=True,
                   choices=['kerr', 'saturable', 'tpa'])
    p.add_argument('--ga0',         type=float, required=True)
    p.add_argument('--noise_scale', type=float, default=0.0)
    p.add_argument('--seed',        type=int,   default=0)
    p.add_argument('--mode',        type=str,   default='sigma_y',
                   choices=['sigma_x', 'sigma_y'])
    p.add_argument('--num_epochs',  type=int,   default=NUM_EPOCHS)
    p.add_argument('--n_s',         type=float, default=None)
    p.add_argument('--tpa_ratio',   type=float, default=None)
    args = p.parse_args()

    params = dict(NL_DEFAULTS[args.nl])
    if args.n_s is not None:
        params['n_s'] = args.n_s
    if args.tpa_ratio is not None:
        params['tpa_ratio'] = args.tpa_ratio

    os.makedirs(OUT_DIR, exist_ok=True)
    run_one(args.nl, args.ga0, args.noise_scale, args.seed, args.mode,
            params, args.num_epochs)


if __name__ == '__main__':
    main()
