# Origin: Corrected_XOR/robustness/plot_robustness_seeds.py (defaults adapted; SI Figs. robustness_training_5seed and robustness_phase_5seed)
"""
plot_robustness_seeds.py — aggregate the 5-seed static robustness study.

Reads results_robustness_seeds/robustness_seed*.npz (one per J0 seed, from
robustness_seed_worker.py) and plots each disorder panel as the MEAN over seeds
with a ±1σ band computed ACROSS SEEDS — i.e. genuine seed-to-seed error bars,
addressing the single-seed caveat of the original figure.

PDFs figures saved in → Figs/ (large text, editable fonts).

Usage
-----
  conda run -n jax-env python plot_robustness_seeds.py
  conda run -n jax-env python plot_robustness_seeds.py --results_dir results_robustness_seeds
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

COL  = {'no noise': '#1f77b4', 'noise': '#d62728'}
YMIN = 1e-6

PANELS = [
    ('detuning',   'sig_J',   r'$\sigma_\delta\,/\,\kappa$',   '(A) Detuning'),
    ('coupling',   'sig_J',   r'$\sigma_J\,/\,\kappa$',         '(B) Coupling'),
    ('kappa',      'sig_rel', r'$\delta\kappa\,/\,\kappa$',     '(C) Loss rate'),
    ('g_disorder', 'sig_rel', r'$\delta g\,/\,g$',              '(D) Nonlinearity'),
    ('crosstalk',  'sig_J',   r'$\sigma_{\rm XT}\,/\,\kappa$',  '(E) Cross-talk'),
    ('thermal',    'n_th_v',  r'$n_{\rm th}$',                  '(F) Thermal noise'),
]


def _panel(ax, xv, agg, base, xlabel, title=None, show_base=True):
    """agg: key -> (mean_over_seeds, std_over_seeds)."""
    for lab in ('no noise', 'noise'):
        m, s = agg[lab]
        col = COL[lab]
        ax.fill_between(xv, np.maximum(m - s, YMIN), m + s, color=col, alpha=0.18)
        ax.loglog(xv, m, '-o', color=col, ms=7, lw=2.6,
                  label=('trained without shot noise' if lab == 'no noise' else 'trained with shot noise'))
    if show_base:
        for lab in ('no noise', 'noise'):
            ax.axhline(base[lab], color=COL[lab], lw=1.4, ls='--', alpha=0.7)
    ax.set_ylim(bottom=YMIN)
    ax.set_xlabel(xlabel)
    ax.grid(False)
    if title:
        ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=os.path.join(_HERE, 'results_robustness_seeds_sc'))
    ap.add_argument('--fig_dir', default=FIGDIR,
                    help='Output directory for figures (default Figs/).')
    args = ap.parse_args()
    figdir = args.fig_dir

    files = sorted(glob.glob(os.path.join(args.results_dir, 'robustness_seed*.npz')))
    if not files:
        print(f"No robustness_seed*.npz in {args.results_dir}")
        return
    ds = [np.load(f) for f in files]
    nseed = len(ds)
    print(f"Aggregating {nseed} seeds: {[int(d['seed']) for d in ds]}")

    os.makedirs(figdir, exist_ok=True)
    plt.rcParams.update({
        'font.size': 19, 'axes.labelsize': 22, 'axes.titlesize': 20,
        'xtick.labelsize': 17, 'ytick.labelsize': 17, 'legend.fontsize': 17,
        'lines.linewidth': 2.6, 'lines.markersize': 8, 'axes.linewidth': 1.3,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'dejavusans',
    })

    # axis grids (identical across seeds)
    grids = {k: ds[0][k] for k in ('sig_J', 'sig_rel', 'n_th_v')}
    has_phase = all('phi_v' in d.files and 'phase_m_c' in d.files for d in ds)
    if has_phase:
        grids['phi_v'] = ds[0]['phi_v']

    # baselines averaged over seeds
    base = {'no noise': float(np.mean([d['loss0_clean'] for d in ds])),
            'noise':    float(np.mean([d['loss0_noisy'] for d in ds]))}

    def aggregate(key):
        """Return {'no noise':(mean,std), 'noise':(mean,std)} across seeds."""
        mc = np.array([d[f'{key}_m_c'] for d in ds])   # (nseed, n_sigma)
        mn = np.array([d[f'{key}_m_n'] for d in ds])
        return {'no noise': (mc.mean(0), mc.std(0)),
                'noise':    (mn.mean(0), mn.std(0))}

    # collective 2x3 panel
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for ax, (key, gk, xl, title) in zip(axes.flatten(), PANELS):
        _panel(ax, grids[gk], aggregate(key), base, xl, title)
        if key in ('detuning', 'g_disorder'):
            ax.set_ylabel('Loss')
    handles = [plt.Line2D([0], [0], color=COL['no noise'], lw=3, label='trained without shot noise'),
               plt.Line2D([0], [0], color=COL['noise'],    lw=3, label='trained with shot noise'),
               plt.Line2D([0], [0], color='gray', lw=1.6, ls='--', label='unperturbed baseline')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'Static disorder robustness — mean $\\pm$ 1$\\sigma$ over {nseed} seeds',
                 fontsize=21, y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(os.path.join(figdir, 'robustness_summary_5seed.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("Saved  Figs/robustness_summary_5seed.pdf")

    # standalone thermal (corrected panel, seed-averaged)
    fig, ax = plt.subplots(figsize=(8, 6))
    _panel(ax, grids['n_th_v'], aggregate('thermal'), base, r'$n_{\rm th}$')
    ax.set_ylabel('Loss'); ax.legend(loc='upper left')
    fig.savefig(os.path.join(figdir, 'robustness_thermal_5seed.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("Saved  Figs/robustness_thermal_5seed.pdf")

    # standalone readout-phase-noise inference panel (if present)
    if has_phase:
        fig, ax = plt.subplots(figsize=(8, 6))
        _panel(ax, grids['phi_v'], aggregate('phase'), base, r'$\sigma_\phi$  [rad]', show_base=False)
        ax.set_ylabel('Loss'); ax.legend(loc='upper left')
        ax2 = ax.twiny(); ax2.set_xscale('log')
        lo, hi = ax.get_xlim(); ax2.set_xlim(np.degrees(lo), np.degrees(hi))
        ax2.set_xlabel(r'$\sigma_\phi$  [deg]')
        fig.savefig(os.path.join(figdir, 'robustness_phase_5seed.pdf'), bbox_inches='tight')
        plt.close(fig)
        print("Saved  robustness_phase_5seed.pdf")

    # training curves (per-seed thin traces + bold seed-mean) 
    if all('lh_clean' in d.files and 'lh_noisy' in d.files for d in ds):
        fig, ax = plt.subplots(figsize=(8, 6))
        for tag, lab in [('lh_clean', 'no noise'), ('lh_noisy', 'noise')]:
            lh = np.array([d[tag] for d in ds])          # (nseed, n_ep)
            ep = np.arange(1, lh.shape[1] + 1)
            ax.semilogy(ep, lh.mean(0), color=COL[lab], lw=2.6,
                        label=('trained without shot noise' if lab == 'no noise' else 'trained with shot noise'))
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(loc='upper right'); ax.grid(False)
        fig.savefig(os.path.join(figdir, 'robustness_training_5seed.pdf'), bbox_inches='tight')
        plt.close(fig)
        print("Saved  robustness_training_5seed.pdf")

    # 10× tolerance thresholds per type, averaged over seeds
    print("\n=== disorder at which mean loss first exceeds 10x baseline ===")
    for key, gk, _, title in PANELS:
        xv = grids[gk]; agg = aggregate(key)
        for lab in ('no noise', 'noise'):
            m = agg[lab][0]; thr = 10 * base[lab]
            idx = np.where(m > thr)[0]
            cross = f"{xv[idx[0]]:.3g}" if len(idx) else ">range"
            print(f"  {title:20s} {lab:9s}: 10x @ {cross}")


if __name__ == '__main__':
    main()
