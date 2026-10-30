"""
plot_cos_phase.py — standalone figure: SB-gradient alignment vs readout phase noise.

Reads results_cos_phase/cos_phase_seed*.npz and plots cos(SB gradient, exact
gradient) at init vs sigma_phi, for clean and noisy dynamics, mean +/- 1 sigma
pooled over seeds x realizations. x-axis log (rad), secondary axis in degrees.

Usage
-----
  conda run -n jax-env python plot_cos_phase.py \
      --results_dir results_cos_phase --fig_dir Figs/g320
"""
import os, glob, argparse, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
COL = {'clean': '#1f77b4', 'noisy': '#d62728'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=os.path.join(_HERE, 'Phase_perport', 'results_cos_phase_perport_rotrec'))
    ap.add_argument('--fig_dir', default=os.path.join(_HERE, 'Figs'))
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.results_dir, 'cos_phase_seed*.npz')))
    if not files:
        print(f"No cos_phase_seed*.npz in {args.results_dir}"); return
    ds = [np.load(f) for f in files]
    phi = ds[0]['phi_v']
    # pool realizations across seeds: shape (n_phi, n_seed*n_real)
    cc = np.concatenate([d['cos_clean'] for d in ds], axis=1)
    cn = np.concatenate([d['cos_noisy'] for d in ds], axis=1)
    print(f"Aggregating {len(ds)} seeds x {ds[0]['cos_clean'].shape[1]} realizations")

    os.makedirs(args.fig_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 19, 'axes.labelsize': 22, 'xtick.labelsize': 17,
                         'ytick.labelsize': 17, 'legend.fontsize': 17, 'lines.linewidth': 2.6,
                         'axes.linewidth': 1.3, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'mathtext.fontset': 'dejavusans'})
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for arr, lab in [(cc, 'clean'), (cn, 'noisy')]:
        m, s = arr.mean(1), arr.std(1)
        ax.fill_between(phi, m - s, m + s, color=COL[lab], alpha=0.18)
        ax.semilogx(phi, m, '-o', color=COL[lab], ms=6,
                    label=('trained without shot noise' if lab == 'clean' else 'trained with shot noise'))
    ax.set_xlabel(r'$\sigma_\phi$  [rad]')
    ax.set_ylabel(r'$\cos(\nabla^{SB_y}_\theta C,\ \nabla_\theta C)$')
    ax.set_ylim(-0.2, 1.02); ax.grid(False); ax.legend(loc='lower left')
    ax2 = ax.twiny(); ax2.set_xscale('log')
    lo, hi = ax.get_xlim(); ax2.set_xlim(np.degrees(lo), np.degrees(hi))
    ax2.set_xlabel(r'$\sigma_\phi$  [deg]')
    fig.tight_layout()
    out = os.path.join(args.fig_dir, 'cos_vs_phase.pdf')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"Saved  {out}")


if __name__ == '__main__':
    main()
