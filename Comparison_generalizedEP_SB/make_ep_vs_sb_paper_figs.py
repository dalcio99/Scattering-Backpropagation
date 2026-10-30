"""
make_ep_vs_sb_paper_figs.py — paper-ready EP-vs-SB figures, styled like
Other_nonlinearities/Paper_figs/make_paper_figs.py (no titles, big fonts,
Wong/Okabe-Ito colours).

Produces (PDF + PNG) in this folder:
  epsb_fig_cos.{pdf,png}        — cos(grad estimate, exact grad) vs ga0
                                  [from ep_vs_sb_data.npz, batched fidelity]
  epsb_fig_loss_ga0.1.{pdf,png} — XOR training loss, ga0 = 0.1 (batched),
                                  log-y / linear-x
  epsb_fig_loss_ga5.{pdf,png}   — XOR training loss, ga0 = 5 (batched),
                                  log-y / linear-x

Label convention (quadrature-basis P notation of the EP section):
  code 'sb_x' (x-quadrature nudge, a-basis sigma_x) -> SB (P1=P2=sigma_z)
  code 'sb_y' (p-quadrature nudge, a-basis sigma_y) -> SB (P1=P2=sigma_x)

  python make_ep_vs_sb_paper_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# colours (same Wong palette as the different-nonlinearities figures)
# ---------------------------------------------------------------------------
COLOR = {'ep':    '#D55E00',   # vermilion / orange
         'sb_x':  '#0072B2',   # blue   -> SB (P=sigma_z)
         'sb_y':  '#009E73',   # green  -> SB (P=sigma_x)
         'exact': 'black'}
LABEL = {'ep':    r'EP ($P_1=P_2=\mathbf{I}_{2N}$)',
         'sb_x':  r'SB ($P_1=P_2=\sigma_z$)',
         'sb_y':  r'SB ($P_1=P_2=\sigma_x$)',
         'exact': r'exact $\nabla_\theta C$'}

# ---------------------------------------------------------------------------
# global style: identical to make_paper_figs.py
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size':        20,
    'axes.labelsize':   26,
    'axes.linewidth':   1.4,
    'xtick.labelsize':  20,
    'ytick.labelsize':  20,
    'legend.fontsize':  22,
    'xtick.major.size': 7,
    'ytick.major.size': 7,
    'xtick.major.width': 1.4,
    'ytick.major.width': 1.4,
    'figure.dpi':       140,
})
LEGEND_KW = dict(frameon=False, loc='best', fontsize=18)


def _save(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'{stem}.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f'saved {stem}.pdf/.png')

# ---------------------------------------------------------------------------
# Figure 1: gradient fidelity vs nonlinearity
# ---------------------------------------------------------------------------
d = np.load(os.path.join(HERE, 'Data', 'ep_vs_sb_data.npz'))
ga0 = np.asarray(d['ga0_list'], dtype=float)
fig, ax = plt.subplots(figsize=(7.6, 5.6))
for key, npz_key in (('sb_x', 'cosA_sb_x'), ('sb_y', 'cosA_sb_y'), ('ep', 'cosA_ep')):
    pts = d[npz_key]                                    # (n_ga0, n_seeds)
    m, sd = pts.mean(axis=1), pts.std(axis=1)
    ax.errorbar(ga0, m, yerr=sd, marker='o', ms=7, lw=2.8, capsize=4,
                color=COLOR[key], label=LABEL[key], zorder=3)
    for i, x in enumerate(ga0):
        ax.plot([x]*pts.shape[1], pts[i], '.', ms=4, color=COLOR[key],
                alpha=0.30, zorder=1)
ax.axhline(0, color='k', lw=0.9, alpha=0.5)
ax.set_xlabel(r'$g\,\langle|a_\mathrm{in}|^2\rangle_\mathrm{avg}/\bar{\kappa}^2$')
ax.set_ylabel(r'$\cos(\nabla_\theta^{\rm est} C,\ \nabla_\theta C)$')
ax.set_ylim(-1.05, 1.05)
ax.legend(frameon=False, loc='lower left', fontsize=18)
ax.grid(alpha=0.25)
_save(fig, 'epsb_fig_cos')

# ---------------------------------------------------------------------------
# Figures 2–3: training loss (batched), log-y / linear-x, means only
# ---------------------------------------------------------------------------
for tag, stem in (('_ga0.1', 'epsb_fig_loss_ga0.1'),
                  ('_ga5',   'epsb_fig_loss_ga5')):
    d = np.load(os.path.join(HERE, 'Data', f'ep_vs_sb_training_data{tag}.npz'),
                allow_pickle=True)
    n_epochs = d['loss_sb_x'].shape[1]
    ep_ax = np.arange(1, n_epochs + 1)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    handles = []
    for key in ('ep', 'sb_x', 'sb_y', 'exact'):
        h = d[f'loss_{key}']
        ls = '--' if key == 'exact' else '-'
        ax.semilogy(ep_ax, np.nanmean(h, axis=0), color=COLOR[key], lw=3.2,
                    ls=ls, zorder=3)
        handles.append(Line2D([0], [0], color=COLOR[key], lw=3.2, ls=ls,
                              label=LABEL[key]))
    ax.axhline(0.0625, color='k', lw=1.0, ls=':', alpha=0.6)   # MSE = 1/16
    ax.set_xlim(0, n_epochs)
    ax.set_xticks(np.arange(0, n_epochs + 1, 200))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend(handles=handles, **LEGEND_KW)
    ax.grid(alpha=0.25)
    _save(fig, stem)
