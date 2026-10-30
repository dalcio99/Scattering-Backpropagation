"""
plot_phase_train.py — standalone figure: does SB still train under readout phase noise?

Reads results_phase_train/*.npz and plots the final CLEAN eval loss (phase-noise-free
performance of the phase-trained network) vs the training-time phase noise sigma_phi,
for clean and noisy dynamics, mean +/- 1 sigma across seeds.

Usage
-----
  conda run -n jax-env python plot_phase_train.py \
      --results_dir results_phase_train --fig_dir PFigs/g320
"""
import os, glob, argparse, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

_HERE = os.path.dirname(os.path.abspath(__file__))
COL = {0: '#1f77b4', 1: '#d62728'}
LAB = {0: 'trained without shot noise', 1: 'trained with shot noise'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=os.path.join(_HERE, 'Phase_perport', 'results_phase_train_perport_rotrec'))
    ap.add_argument('--fig_dir', default=os.path.join(_HERE, 'Figs'))
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.results_dir, 'phase_train_*.npz')))
    if not files:
        print(f"No phase_train_*.npz in {args.results_dir}"); return
    # group: {(noise, batch) -> {sigma_phi -> [clean_eval_loss over seeds]}}
    data = {}
    for f in files:
        d = np.load(f)
        ns = int(d['noise']); bt = int(d['batch']) if 'batch' in d.files else 0
        sp = float(d['sigma_phi'])
        data.setdefault((ns, bt), {}).setdefault(sp, []).append(float(d['clean_eval_loss']))

    os.makedirs(args.fig_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 19, 'axes.labelsize': 22, 'xtick.labelsize': 17,
                         'ytick.labelsize': 17, 'legend.fontsize': 17, 'lines.linewidth': 2.6,
                         'axes.linewidth': 1.3, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'mathtext.fontset': 'dejavusans'})
    fig, ax = plt.subplots(figsize=(8.5, 6))
    STYLE = {0: ('-', 'online'), 1: ('--', 'batch')}
    print("=== final clean eval loss vs training-time phase noise ===")
    for (ns, bt) in sorted(data):
        ls, mode = STYLE[bt]
        sps = np.array(sorted(data[(ns, bt)].keys()))
        m = np.array([np.mean(data[(ns, bt)][s]) for s in sps])
        s = np.array([np.std(data[(ns, bt)][s]) for s in sps])
        x = np.where(sps <= 0, 1e-3, sps)   # sigma_phi=0 shown at 1e-3 on log axis
        ax.fill_between(x, np.maximum(m - s, 1e-8), m + s, color=COL[ns], alpha=0.12)
        ax.loglog(x, m, ls, marker='o', color=COL[ns], ms=6,
                  label=f'{LAB[ns]} ({mode})')
        print(f"  {LAB[ns]} ({mode}):")
        for sp, mm in zip(sps, m):
            print(f"    sigma_phi={sp:.3f}: clean eval loss = {mm:.3e}")
    ax.set_xlabel(r'$\sigma_\phi$  [rad]')
    ax.set_ylabel('Loss')
    ax.grid(False); ax.legend(loc='lower right')

    # y-axis: minor ticks every decade (1/10 spacing), labelled majors at even powers of 10
    ax.yaxis.set_major_locator(LogLocator(base=100.0))                 # 10^0, 10^-2, 10^-4, ...
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,), numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # secondary top axis: sigma_phi in degrees
    ax2 = ax.twiny(); ax2.set_xscale('log')
    lo, hi = ax.get_xlim(); ax2.set_xlim(np.degrees(lo), np.degrees(hi))
    ax2.set_xlabel(r'$\sigma_\phi$  [deg]')

    fig.tight_layout()
    out = os.path.join(args.fig_dir, 'phase_train.pdf')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"Saved  {out}")


if __name__ == '__main__':
    main()
