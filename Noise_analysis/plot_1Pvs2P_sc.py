"""
plot_1Pvs2P_sc.py — combined measurement-time figure, semiclassical convention.

Recreates the SI figure `1Pvs2P_fig_sigma_y_tsamp_noise` (fig:comparison_T_m):
MSE loss histories for different measurement times T_m, sigma_y, noise = 1;
left panel: one linear-response experiment (2-phase SB, results_tsamp_sc);
right panel: two linear-response experiments (3-phase/symmetric nudging,
results_tsamp_sc_3phase). Mean over seeds; thin lines = single seeds.

New vs the old figure: dashed horizontal lines mark the predicted
measurement-noise loss floor Var[y] = 1/(4 T_m) of the record-based estimator
(exact for the lossless network at g = 0; the trained networks should saturate
near it).

Run:  /opt/miniconda3/envs/jax-env/bin/python plot_1Pvs2P_sc.py
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'Figures')
_CMAP = plt.cm.viridis


def load(results_dir):
    rows = {}
    for f in sorted(glob.glob(os.path.join(HERE, results_dir,
                                           'sigma_y_*noise1.0*tsamp*.npz'))):
        z = np.load(f, allow_pickle=True)
        t = float(z['t_sample'])
        rows.setdefault(t, []).append(np.asarray(z['loss_history']))
    return {t: np.array(v) for t, v in sorted(rows.items())}


PAPER = True   # paper style: no titles/grid/seed traces, big fonts

if PAPER:
    plt.rcParams.update({
        'font.size':        18,
        'axes.labelsize':   24,
        'axes.linewidth':   1.4,
        'xtick.labelsize':  18,
        'ytick.labelsize':  18,
        'legend.fontsize':  17,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.3,
        'ytick.major.width': 1.3,
    })


def panel(ax, data, title):
    tsamps = sorted(data.keys())
    for i, t in enumerate(tsamps):
        c = _CMAP(0.10 + 0.80 * i / max(len(tsamps) - 1, 1))
        lh = data[t]
        ep = np.arange(1, lh.shape[1] + 1)
        if not PAPER:
            for row in lh:
                ax.semilogy(ep, row, color=c, alpha=0.12, lw=0.5)
        ax.semilogy(ep, lh.mean(axis=0), color=c, lw=2.2,
                    label=rf'$\bar\kappa T_m={t:g}$')
        ax.axhline(1.0 / (4.0 * t), color=c, ls='--', lw=1.2, alpha=0.75)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim(0, lh.shape[1])
    if not PAPER:
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    ax.legend(loc='upper right', framealpha=0.9,
              handlelength=1.4, borderaxespad=0.3)


def main():
    os.makedirs(OUT, exist_ok=True)
    d1 = load(os.path.join('Data', 'results_tsamp_sc'))
    d2 = load(os.path.join('Data', 'results_tsamp_sc_3phase'))
    print('2-phase:', {t: v.shape[0] for t, v in d1.items()})
    print('3-phase:', {t: v.shape[0] for t, v in d2.items()})

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    panel(axes[0], d1, 'one linear-response experiment (SB)')
    panel(axes[1], d2, r'two linear-response experiments (SB$^{\pm}$)')
    axes[1].set_ylabel('')
    if not PAPER:
        fig.suptitle(r'[$\sigma_y$, noise=1]  dashed: predicted floor'
                     r' $1/(4T_m)$', fontsize=10)
    fig.tight_layout()
    fpath = os.path.join(OUT, '1Pvs2P_fig_sigma_y_tsamp_noise_sc.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    fig.savefig(fpath.replace('.pdf', '.png'), dpi=220, bbox_inches='tight')
    print('saved', fpath)

    # floor check table
    print(f"\n{'T_m':>6} {'floor(2p,last25)':>17} {'floor(3p,last25)':>17} "
          f"{'pred 1/(4Tm)':>13}")
    for t in sorted(d1.keys()):
        f1 = float(np.mean([r[-25:].mean() for r in d1[t]]))
        f2 = float(np.mean([r[-25:].mean() for r in d2.get(t, d1[t])]))
        print(f"{t:>6g} {f1:>17.3e} {f2:>17.3e} {1/(4*t):>13.3e}")


if __name__ == '__main__':
    main()
