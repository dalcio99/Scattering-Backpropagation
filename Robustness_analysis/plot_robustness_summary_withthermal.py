"""plot_robustness_summary_withthermal.py — 6-panel variant: the 5 disorder
panels of robustness_summary_5seed_nothermal (a0=160) PLUS a thermal panel read
from a separately-produced thermal-data npz (e.g. the high-power a0=1600 run).

NOTE: if the thermal npz is the a0=1600 run, this figure MIXES operating points
(disorder panels at a0=160, thermal at a0=1600) — the thermal panel title
annotates its power so the mismatch is explicit.

Same fixed axes/style as plot_robustness_summary_variant.py:
  disorder panels x in [1e-3, 1e-1]; ALL panels y in [1e-6, 1e1]; thermal panel
  keeps its own n_th x-range.

Usage
-----
  conda run -n jax-env python plot_robustness_summary_withthermal.py \
      --results_dir <robustness_seed*.npz dir> \
      --thermal_npz <thermal_*_data.npz> \
      --thermal_label '$|a_{\rm in}|^2=1600$' \
      --fig_dir <out dir> --out_name robustness_summary_5seed_with_hp_thermal.pdf
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
COL = {'no noise': '#1f77b4', 'noise': '#d62728'}
YMIN, YMAX = 1e-6, 1e1
XMIN, XMAX = 1e-3, 1e-1

DISORDER = [
    ('detuning',   'sig_J',   r'$\sigma_\delta\,/\,\bar{\kappa}$',       r'$\mathbf{a}$  Detuning'),
    ('coupling',   'sig_J',   r'$\sigma_J\,/\,\bar{\kappa}$',            r'$\mathbf{b}$  Coupling'),
    ('kappa',      'sig_rel', r'$\delta\bar{\kappa}\,/\,\bar{\kappa}$',  r'$\mathbf{c}$  Loss rate'),
    ('g_disorder', 'sig_rel', r'$\delta g\,/\,g$',                       r'$\mathbf{d}$  Nonlinearity'),
    ('crosstalk',  'sig_J',   r'$\sigma_{\rm XT}\,/\,\bar{\kappa}$',     r'$\mathbf{e}$  Cross-talk'),
]


def _panel(ax, xv, agg, base, xlabel, title, xlim):
    for lab in ('no noise', 'noise'):
        m, s = agg[lab]; col = COL[lab]
        ax.fill_between(xv, np.maximum(m - s, YMIN), m + s, color=col, alpha=0.18)
        ax.loglog(xv, m, '-o', color=col, ms=7, lw=2.6,
                  label=('trained without shot noise' if lab == 'no noise' else 'trained with shot noise'))
    for lab in ('no noise', 'noise'):
        ax.axhline(base[lab], color=COL[lab], lw=1.4, ls='--', alpha=0.7)
    ax.set_xlim(*xlim); ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel(xlabel); ax.grid(False); ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=os.path.join(_HERE, 'results_robustness_seeds_sc'))
    ap.add_argument('--thermal_npz', default=os.path.join(_HERE, 'results_thermal_noise_sc', 'thermal_a0_1600_5seed_data.npz'))
    ap.add_argument('--thermal_label', default='',
                    help='annotation appended to the thermal panel title, e.g. power')
    ap.add_argument('--fig_dir', default=FIGDIR)
    ap.add_argument('--out_name', default='robustness_summary_5seed_with_hp_thermal.pdf')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, 'robustness_seed*.npz')))
    if not files:
        print(f"No robustness_seed*.npz in {args.results_dir}"); return
    ds = [np.load(f) for f in files]
    nseed = len(ds)
    print(f"Disorder panels: {nseed} seeds {[int(d['seed']) for d in ds]}")

    grids = {k: ds[0][k] for k in ('sig_J', 'sig_rel')}
    base = {'no noise': float(np.mean([d['loss0_clean'] for d in ds])),
            'noise':    float(np.mean([d['loss0_noisy'] for d in ds]))}

    def aggregate(key):
        mc = np.array([d[f'{key}_m_c'] for d in ds])
        mn = np.array([d[f'{key}_m_n'] for d in ds])
        return {'no noise': (mc.mean(0), mc.std(0)), 'noise': (mn.mean(0), mn.std(0))}

    # thermal panel from the separate npz (curve_clean/curve_noisy: (nseed, n_nth))
    th = np.load(args.thermal_npz)
    NTH = th['NTH']; cc = th['curve_clean']; cn = th['curve_noisy']
    print(f"Thermal panel: {cc.shape[0]} seeds, {len(NTH)} n_th points, "
          f"from {os.path.basename(args.thermal_npz)}")
    thermal_agg = {'no noise': (cc.mean(0), cc.std(0)), 'noise': (cn.mean(0), cn.std(0))}
    thermal_base = {'no noise': float(np.mean(th['base_clean'])),
                    'noise':    float(np.mean(th['base_noisy']))}
    th_title = r'$\mathbf{f}$  Thermal noise'   # power annotation goes in the caption

    os.makedirs(args.fig_dir, exist_ok=True)
    plt.rcParams.update({
        'font.size': 19, 'axes.labelsize': 22, 'axes.titlesize': 20,
        'xtick.labelsize': 17, 'ytick.labelsize': 17, 'legend.fontsize': 17,
        'lines.linewidth': 2.6, 'lines.markersize': 8, 'axes.linewidth': 1.3,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'dejavusans',
    })
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    flat = axes.flatten()
    for ax, (key, gk, xl, title) in zip(flat[:5], DISORDER):
        _panel(ax, grids[gk], aggregate(key), base, xl, title, (XMIN, XMAX))
        if key in ('detuning', 'g_disorder'):
            ax.set_ylabel('Loss')
    _panel(flat[5], NTH, thermal_agg, thermal_base, r'$n_{\rm th}$', th_title,
           (NTH.min(), NTH.max()))

    handles = [plt.Line2D([0], [0], color=COL['no noise'], lw=3, label='trained without shot noise'),
               plt.Line2D([0], [0], color=COL['noise'],    lw=3, label='trained with shot noise'),
               plt.Line2D([0], [0], color='gray', lw=1.6, ls='--', label='unperturbed baseline')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(args.fig_dir, args.out_name)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved  {out}")


if __name__ == '__main__':
    main()
