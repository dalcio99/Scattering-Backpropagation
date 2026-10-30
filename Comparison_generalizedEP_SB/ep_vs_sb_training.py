"""
ep_vs_sb_training.py — does vector-field EP [1]
actually TRAIN our port-Hamiltonian systems? (deterministic ODEs)

Companion to ep_vs_sb.py (gradient fidelity): same setting (N = 10 fully
connected self-Kerr, XOR, a0_in_sq = 160, out node 3, noiseless), fixed
effective nonlinearity g<|a_in|^2>_avg/kbar^2 = 1.0 (g = 1/160).

Arms (independent training runs from the SAME J init per seed):
  sb_x      : SB update, U = sigma_x        (eta = 5e-4)
  ep        : EP (P = 1) update       (eta = 5e-4)
  ep_lr_hi  : EP, eta = 5e-3          (rule out "learning rate too small")
  ep_lr_lo  : EP, eta = 5e-5          (rule out "learning rate too large")
  exact     : exact IFT gradient            (eta = 5e-4, reference)

sb_x and ep use the IDENTICAL physical protocol (same +beta*err x-quadrature
port nudge); they differ only by one conjugation in the contraction, exactly as
in ep_vs_sb.py. Batched updates over the 4 XOR patterns; beta = 1e-3;
800 epochs; 8 seeds run in parallel (multiprocessing).

A run is hard-stopped (loss -> NaN afterwards) if the state diverges.

Run:  /opt/miniconda3/envs/jax-env/bin/python ep_vs_sb_training.py
(imports sc_core.py / gradient_worker_sc.py from the parent folder)

Citations:
[1] Scellier, Benjamin, et al. "Generalization of equilibrium propagation to vector 
                                field dynamics." arXiv preprint arXiv:1808.04873 (2018).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'Noise_analysis'))  # shared modules (sc_core, gradient_worker_sc)
from sc_core import simulate_record, XOR_INPUTS, XOR_TARGETS
from gradient_worker_sc import ift_gradient_batch, _make_J, _ss_noiseless, \
                               T_SAMPLE_NOISELESS
# ---------------------------------------------------------------------------
# Setting (identical to ep_vs_sb.py)
# ---------------------------------------------------------------------------

N          = 10
a0_in_sq   = 160.0
amplitude  = float(np.sqrt(a0_in_sq * N))
out_idx    = 2
KAPPA      = np.ones(N)
BETA       = 1e-3
T_TRANS, T_SS, DT = 40.0, 20.0, 0.05

# env overrides read at MODULE level so multiprocessing spawn-workers (which
# re-import this module) see the same values as the parent
GA0        = float(os.environ.get('EPVS_GA0', 1.0))
G_NL       = GA0 / a0_in_sq          # g = GA0/160
NUM_EPOCHS = int(os.environ.get('EPVS_EPOCHS', 800))
SEEDS      = list(range(int(os.environ.get('EPVS_SEEDS', 8))))
ETA_BASE   = 5e-4
MODE       = os.environ.get('EPVS_MODE', 'batched')   # 'batched' | 'online'
TAG        = f"_ga{GA0:g}" + ("_online" if MODE == 'online' else "")

_ALL_ARMS = {'sb_x':     ('sb_x', ETA_BASE),
             'sb_y':     ('sb_y', ETA_BASE),
             'ep':       ('ep',   ETA_BASE),
             'ep_lr_hi': ('ep',   5e-3),
             'ep_lr_lo': ('ep',   5e-5),
             'exact':    ('exact', ETA_BASE)}
_arm_names = os.environ.get('EPVS_ARMS', 'sb_x,ep,ep_lr_hi,ep_lr_lo,exact').split(',')
ARMS = [(a, *_ALL_ARMS[a]) for a in _arm_names]

DIVERGE_ABS = 1e6


def det_ss(J, a_in, rng):
    """Simulate the (deterministic) dynamics and return steady-state"""
    return simulate_record(J, G_NL, KAPPA, a_in, None, T_TRANS, T_SS, DT,
                           0.0, a0=None, rng=rng).a_avg


def estimate_grad_and_loss(J, method, rng):
    """Batched gradient estimate + batched MSE at the current J.
    method: 'sb_x' | 'ep' | 'exact'. Returns (grad, loss) or (None, None) on
    divergence."""
    sqk = np.sqrt(KAPPA)
    grad = np.zeros((N, N))
    loss = 0.0
    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex)
        a_in[:2] = amplitude * inp
        a_ss = det_ss(J, a_in, rng)
        if not np.all(np.isfinite(a_ss)) or np.max(np.abs(a_ss)) > DIVERGE_ABS:
            return None, None
        y = (a_in + sqk * a_ss)[out_idx].real
        e = y - tgt
        loss += e**2

        if method == 'exact':
            continue                       # gradient computed once, below

        da = np.zeros(N, dtype=complex)
        if method == 'sb_y':
            da[out_idx] = -1j * BETA * e   # p-quadrature port nudge (sigma_y)
        else:
            da[out_idx] = BETA * e         # x-quadrature port nudge (sb_x / EP)
        a_ss_p = det_ss(J, a_in + da, rng)
        if not np.all(np.isfinite(a_ss_p)) or np.max(np.abs(a_ss_p)) > DIVERGE_ABS:
            return None, None
        d = (a_ss_p - a_ss) / BETA

        if method == 'sb_x':
            G = -2.0 * np.imag(np.outer(d, a_ss))          # transpose pairing
        elif method == 'sb_y':
            G = -2.0 * np.real(np.outer(d, a_ss))          # sigma_y pairing
        else:                                              # EP
            G = +2.0 * np.imag(np.outer(d, a_ss.conj()))   # Hermitian pairing
        grad += G + G.T - np.diag(np.diag(G))

    loss /= len(XOR_INPUTS)
    if method == 'exact':
        grad = ift_gradient_batch(J, G_NL, KAPPA, DT, seed=0)
    else:
        grad /= len(XOR_INPUTS)
    return grad, loss


def ift_gradient_single(J, a_in, tgt, rng):
    """Per-pattern exact IFT gradient (self-Kerr, target_scale = 1), identical
    to one term of gradient_worker_sc.ift_gradient_batch (no /4)."""
    kap = KAPPA
    a_ss = _ss_noiseless(J, G_NL, kap, a_in, T_SAMPLE_NOISELESS, DT, rng)
    u, v = a_ss.real, a_ss.imag
    y = (a_in + np.sqrt(kap) * a_ss)[out_idx].real
    err = y - tgt
    g = G_NL
    A = np.diag(-kap/2.0 + 2.0*g*u*v)
    B = J + np.diag(g*(u**2 + 3.0*v**2))
    C = -J - np.diag(g*(3.0*u**2 + v**2))
    D = np.diag(-kap/2.0 - 2.0*g*u*v)
    Jac = np.block([[A, B], [C, D]])
    e_L = np.zeros(2*N)
    e_L[out_idx] = 2.0 * err * np.sqrt(kap[out_idx])
    lam = np.linalg.solve(Jac.T, e_L)
    lam_u, lam_v = lam[:N], lam[N:]
    G = (-(np.outer(lam_u, v) + np.outer(v, lam_u))
         + (np.outer(lam_v, u) + np.outer(u, lam_v)))
    G[np.diag_indices_from(G)] *= 0.5
    return G, err


def pattern_grad(J, method, a_in, a_ss, e, rng):
    """Symmetrized per-pattern gradient estimate (no batching /4) for the
    port-nudge methods. Returns None on divergence."""
    da = np.zeros(N, dtype=complex)
    if method == 'sb_y':
        da[out_idx] = -1j * BETA * e
    else:
        da[out_idx] = BETA * e
    a_ss_p = det_ss(J, a_in + da, rng)
    if not np.all(np.isfinite(a_ss_p)) or np.max(np.abs(a_ss_p)) > DIVERGE_ABS:
        return None
    d = (a_ss_p - a_ss) / BETA
    if method == 'sb_x':
        G = -2.0 * np.imag(np.outer(d, a_ss))
    elif method == 'sb_y':
        G = -2.0 * np.real(np.outer(d, a_ss))
    else:                                              # EP
        G = +2.0 * np.imag(np.outer(d, a_ss.conj()))
    return G + G.T - np.diag(np.diag(G))


def epoch_online(J, method, eta, rng):
    """One ONLINE epoch: per-pattern update, loss recorded from each free
    phase as encountered. Returns (J, loss) or (None, None) on divergence."""
    sqk = np.sqrt(KAPPA)
    loss = 0.0
    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex)
        a_in[:2] = amplitude * inp
        if method == 'exact':
            a_chk = det_ss(J, a_in, rng)
            if not np.all(np.isfinite(a_chk)) or np.max(np.abs(a_chk)) > DIVERGE_ABS:
                return None, None
            G, e = ift_gradient_single(J, a_in, tgt, rng)
        else:
            a_ss = det_ss(J, a_in, rng)
            if not np.all(np.isfinite(a_ss)) or np.max(np.abs(a_ss)) > DIVERGE_ABS:
                return None, None
            e = (a_in + sqk * a_ss)[out_idx].real - tgt
            G = pattern_grad(J, method, a_in, a_ss, e, rng)
            if G is None:
                return None, None
        loss += e**2
        J = J - eta * G
    return J, loss / len(XOR_INPUTS)


def run_seed(seed):
    out = {}
    for arm_name, method, eta in ARMS:
        rng = np.random.default_rng(seed)
        J = _make_J(seed).copy()
        hist = np.full(NUM_EPOCHS, np.nan)
        for ep_i in range(NUM_EPOCHS):
            if MODE == 'online':
                J_new, loss = epoch_online(J, method, eta, rng)
                if J_new is None:
                    break                   # diverged: NaN from here on
                hist[ep_i] = loss
                J = J_new
            else:
                grad, loss = estimate_grad_and_loss(J, method, rng)
                if grad is None:
                    break                   # diverged: NaN from here on
                hist[ep_i] = loss
                J = J - eta * grad
        out[arm_name] = hist
        last = hist[~np.isnan(hist)]
        print(f"  seed {seed}  {arm_name:9s} eta={eta:g}  "
              f"final={last[-1] if len(last) else float('nan'):.3e}  "
              f"epochs_run={len(last)}", flush=True)
    return seed, out


def main():
    print(f"EP-vs-SB training: ga0={GA0} (g={G_NL:g}), beta={BETA}, "
          f"{NUM_EPOCHS} epochs, seeds={SEEDS}, mode={MODE}")
    with Pool(processes=min(len(SEEDS), os.cpu_count() or 4)) as pool:
        results = dict(pool.map(run_seed, SEEDS))

    hists = {arm: np.array([results[s][arm] for s in SEEDS])
             for arm, _, _ in ARMS}
    np.savez(os.path.join(HERE, 'Data', f'ep_vs_sb_training_data{TAG}.npz'),
             seeds=SEEDS, ga0=GA0, beta=BETA, eta_base=ETA_BASE,
             arms=[a for a, _, _ in ARMS],
             **{f'loss_{a}': hists[a] for a, _, _ in ARMS})

    # summary
    print("\nSummary (per arm): median final loss | seeds with MSE<1/16 | <1e-2")
    for arm, _, eta in ARMS:
        h = hists[arm]
        finals = []
        for row in h:
            v = row[~np.isnan(row)]
            finals.append(v[-1] if len(v) else np.nan)
        finals = np.array(finals)
        ok = np.isfinite(finals)
        n_stopped = sum(int(np.isnan(row).any()) for row in h)   # hard-stopped early
        print(f"  {arm:9s} eta={eta:g}: median={np.nanmedian(finals):.3e}  "
              f"<1/16: {(finals[ok] < 0.0625).sum()}/{len(SEEDS)}  "
              f"<1e-2: {(finals[ok] < 1e-2).sum()}/{len(SEEDS)}  "
              f"stopped-early (diverged): {n_stopped}")

    # figure
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11})
    colors = {'sb_x': '#2ca02c', 'sb_y': '#17becf', 'ep': '#d62728',
              'ep_lr_hi': '#ff7f0e', 'ep_lr_lo': '#9467bd', 'exact': '#1f77b4'}
    labels = {'sb_x': r'SB, $U=\sigma_x$ ($\eta=5\!\times\!10^{-4}$)',
              'sb_y': r'SB, $U=\sigma_y$ ($\eta=5\!\times\!10^{-4}$)',
              'ep': r'EP ($\eta=5\!\times\!10^{-4}$)',
              'ep_lr_hi': r'EP ($\eta=5\!\times\!10^{-3}$)',
              'ep_lr_lo': r'EP ($\eta=5\!\times\!10^{-5}$)',
              'exact': r'exact $\nabla_\theta C$ ($\eta=5\!\times\!10^{-4}$)'}
    epochs = np.arange(1, NUM_EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for arm, _, _ in ARMS:
        h = hists[arm]
        for row in h:
            ax.plot(epochs, row, color=colors[arm], lw=0.5, alpha=0.18)
        ax.plot(epochs, np.nanmean(h, axis=0), color=colors[arm], lw=2.0,
                label=labels[arm])
    ax.axhline(0.0625, color='k', lw=0.7, ls=':', alpha=0.6)
    ax.text(1.15, 0.070, 'MSE = 1/16', fontsize=7.5, alpha=0.7)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1, NUM_EPOCHS)
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    _mode_lbl = ', online updates' if MODE == 'online' else ''
    ax.set_title(rf'XOR training, $g\langle|a_\mathrm{{in}}|^2\rangle_\mathrm{{avg}}'
                 rf'/\bar\kappa^2 = {GA0:g}$   ({len(SEEDS)} seeds{_mode_lbl})',
                 fontsize=10)
    ax.legend(fontsize=7.5, loc='best', framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'fig_ep_vs_sb_training{TAG}.{ext}'),
                    bbox_inches='tight', dpi=250)
    print(f"\nsaved fig_ep_vs_sb_training{TAG}.pdf/.png + ep_vs_sb_training_data{TAG}.npz")


if __name__ == '__main__':
    main()
