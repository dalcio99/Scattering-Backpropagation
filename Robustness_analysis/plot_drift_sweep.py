"""
plot_drift_sweep.py — aggregate the drift-rate sweep and locate the drift level
at which the steady-state MSE crosses the 1e-2 and 1e-1 thresholds.

Reads results_drift_sweep/*.npz (produced by drift_worker.py), averages over
seeds for each (arm, sigma_step), and plots mean steady-state MSE vs the drift
rate sigma_step [κ/epoch]. Crossings of 1e-2 and 1e-1 are found by linear
interpolation in log-log space and reported per arm.

Also emits PDF figures into Figs/.

Usage
-----
  conda run -n jax-env python plot_drift_sweep.py
  conda run -n jax-env python plot_drift_sweep.py --results_dir results_drift_sweep
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(_HERE, 'Figs')

ARM_STYLE = {
    'noisy':  ('trained with shot noise (online)',    '#d62728'),
    'clean':  ('trained without shot noise (online)', '#1f77b4'),
    'frozen': ('frozen (no online learning)',    '#7f7f7f'),
}


def _load(results_dir):
    """Return dict: arm -> {sigma_step -> list of mse_mean over seeds}."""
    data = {}
    for f in glob.glob(os.path.join(results_dir, '*.npz')):
        d = np.load(f, allow_pickle=True)
        arm = str(d['arm'])
        ss  = float(d['sigma_step'])
        data.setdefault(arm, {}).setdefault(ss, []).append(float(d['mse_mean']))
    return data


def _crossing(x, y, level):
    """First x where increasing y crosses `level`, via log–log interpolation."""
    lx, ly, ll = np.log10(x), np.log10(y), np.log10(level)
    for i in range(len(y) - 1):
        if (ly[i] - ll) <= 0 <= (ly[i + 1] - ll):
            t = (ll - ly[i]) / (ly[i + 1] - ly[i])
            return 10 ** (lx[i] + t * (lx[i + 1] - lx[i]))
    if ly[0] > ll:
        return x[0]          # already above at the smallest rate
    return None              # never crosses within the swept range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=os.path.join(_HERE, 'results_drift_sweep_sc'))
    ap.add_argument('--num_epochs',  type=int, default=200,
                    help='For converting sigma_step → accumulated Σ on the top axis.')
    ap.add_argument('--Q', type=float, default=1e5,
                    help='Ring quality factor for the physical (GHz) top axis.')
    ap.add_argument('--fig_dir', default=FIGDIR,
                    help='Output directory for the figure (default Figs/).')
    args = ap.parse_args()
    figdir = args.fig_dir

    os.makedirs(figdir, exist_ok=True)
    data = _load(args.results_dir)
    if not data:
        print(f"No .npz found in {args.results_dir}")
        return

    plt.rcParams.update({
        'font.size': 19, 'axes.labelsize': 22, 'axes.titlesize': 20,
        'xtick.labelsize': 17, 'ytick.labelsize': 17, 'legend.fontsize': 16,
        'lines.linewidth': 2.6, 'lines.markersize': 8, 'axes.linewidth': 1.3,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'dejavusans',
    })

    fig, ax = plt.subplots(figsize=(9, 6.5))
    print("\n=== Drift-rate thresholds (steady-state mean MSE) ===")
    for arm in ['clean', 'noisy', 'frozen']:
        if arm not in data:
            continue
        label, col = ARM_STYLE[arm]
        sigmas = np.array(sorted(data[arm].keys()))
        means  = np.array([np.mean(data[arm][s]) for s in sigmas])
        stds   = np.array([np.std(data[arm][s])  for s in sigmas])
        ax.fill_between(sigmas, np.maximum(means - stds, 1e-6), means + stds,
                        color=col, alpha=0.15)
        ax.loglog(sigmas, means, '-o', color=col, label=label)

        c2 = _crossing(sigmas, means, 1e-2)
        c1 = _crossing(sigmas, means, 1e-1)
        c2s = f"{c2:.3g}" if c2 else "  >range"
        c1s = f"{c1:.3g}" if c1 else "  >range"
        print(f"  {arm:7s}:  MSE=1e-2 at sigma_step≈{c2s} κ/epoch"
              f"   |  MSE=1e-1 at sigma_step≈{c1s} κ/epoch")

    for lvl, txt in [(1e-2, r'MSE $=10^{-2}$'), (1e-1, r'MSE $=10^{-1}$')]:
        ax.axhline(lvl, color='k', lw=1.2, ls='--', alpha=0.55)
        ax.text(ax.get_xlim()[0], lvl * 1.15, txt, fontsize=14, alpha=0.7)

    ax.set_xlabel(r'drift rate  $\sigma_{\rm step}$  [$\bar{\kappa}$/epoch]')
    ax.set_ylabel('Loss')
    ax.grid(False)
    ax.legend(loc='lower right')

    # ── secondary top axis: accumulated detuning Σ over the run, in GHz ─────────
    # Σ = σ_step·√N_ep  (κ units) → ×(κ/2π); κ/2π = 193 THz / Q.
    factor = np.sqrt(args.num_epochs)                # Sigma[kbar] = sigma_step * sqrt(N_ep)
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    lo, hi = ax.get_xlim()
    ax2.set_xlim(lo * factor, hi * factor)
    ax2.set_xlabel(rf'accumulated detuning over {args.num_epochs} epochs  [$\bar{{\kappa}}$]',
                   fontsize=17)
    ax2.tick_params(labelsize=15)

    fig.tight_layout()
    out = os.path.join(figdir, 'drift_rate_thresholds.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved  Figs/drift_rate_thresholds.pdf")


if __name__ == '__main__':
    main()
