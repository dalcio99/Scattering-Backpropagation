"""
plot_drift_during_training.py — illustrative "online training under drift" figure.

Rebuilds the drift_during_training panel from a drift-RATE sweep directory (one
npz per arm/sigma_step/seed, each storing per-epoch eval_lh and drift_trace).
Picks ONE representative drift rate sigma_step and shows, over epochs:
  top    — eval XOR loss for the three arms (noisy / clean / frozen), mean +/- 1sigma
           across seeds;
  bottom — the accumulated common-mode detuning random walk (per-seed traces).

Usage
-----
  conda run -n jax-env python plot_drift_during_training.py \
      --results_dir results_drift_sweep_sc \
      --sigma_step 0.009703 --fig_dir Figs/drift \
      --out_name drift_during_training_s009703.pdf
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

ARMS = [('noisy',  'trained with shot noise (online)',    '#d62728'),
        ('clean',  'trained without shot noise (online)', '#1f77b4'),
        ('frozen', 'frozen (no online learning)',         '#7f7f7f')]


def _load(results_dir, arm, sigma_step, tol=1e-3):
    """Return (eval_lh stack [n_seed, n_ep], drift_trace stack) for one arm/sigma."""
    evs, drs = [], []
    for f in sorted(glob.glob(os.path.join(results_dir, f'{arm}_*.npz'))):
        d = np.load(f)
        if abs(float(d['sigma_step']) - sigma_step) <= tol * max(sigma_step, 1e-9):
            evs.append(d['eval_lh']); drs.append(d['drift_trace'])
    return np.array(evs), np.array(drs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=os.path.join(_HERE, 'results_drift_sweep_sc'))
    ap.add_argument('--sigma_step', type=float, default=0.009703,
                    help='Representative drift rate [kappa/epoch] to illustrate.')
    ap.add_argument('--fig_dir', default=os.path.join(_HERE, 'Figs'))
    ap.add_argument('--out_name', default='drift_during_training.pdf')
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 19, 'axes.labelsize': 22, 'axes.titlesize': 19,
        'xtick.labelsize': 17, 'ytick.labelsize': 17, 'legend.fontsize': 15,
        'lines.linewidth': 2.6, 'axes.linewidth': 1.3,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'dejavusans',
    })

    fig, (ax, axd) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})

    drift_stack = None
    for arm, label, col in ARMS:
        ev, dr = _load(args.results_dir, arm, args.sigma_step)
        if ev.size == 0:
            print(f"  no data for arm={arm} at sigma_step={args.sigma_step}")
            continue
        ep = np.arange(1, ev.shape[1] + 1)
        for tr in ev:                                  # thin per-seed traces
            ax.semilogy(ep, np.maximum(tr, 1e-6), color=col, lw=0.7, alpha=0.30)
        ax.semilogy(ep, ev.mean(0), color=col, lw=3.0, label=label)   # bold mean
        if arm == 'noisy':
            drift_stack = dr

    ax.set_ylabel('Loss')
    ax.set_title(r'Online training under common-mode detuning drift')
    ax.grid(False)
    ax.legend(loc='lower right')

    # bottom: accumulated common-mode drift (per-seed random walks)
    if drift_stack is not None:
        ep = np.arange(1, drift_stack.shape[1] + 1)
        for tr in drift_stack:
            axd.plot(ep, tr, color='#d62728', lw=1.0, alpha=0.35)
        axd.plot(ep, drift_stack.mean(0), color='#d62728', lw=2.4, label='mean')
    axd.axhline(0.0, color='k', lw=0.9, ls='-', alpha=0.4)
    axd.set_ylabel(r'accum. drift [$\bar{\kappa}$]')
    axd.set_xlabel('Epoch')
    axd.grid(False)

    fig.tight_layout()
    out = os.path.join(args.fig_dir, args.out_name)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved  {out}")


if __name__ == '__main__':
    main()
