"""
make_paper_figs.py — paper-ready figures for the "other nonlinearities" study,
SEMICLASSICAL convention (sc data).

Produces three figures (PDF + PNG) in this folder:
  fig_loss_noiseless.{pdf,png}  — XOR loss vs epoch, all nonlinearities, NO noise
  fig_loss_noisy.{pdf,png}      — XOR loss vs epoch, all nonlinearities, WITH noise
  fig_photon.{pdf,png}          — <n>_avg vs epoch, colour = nonlinearity,
                                  shade = noiseless (dark) / noisy (light)

Loss = mean over seeds (thick), individual seeds faint.  Photon = mean over seeds.
No titles, big fonts, minimal legend (nonlinearity names only).

  python make_paper_figs.py [--results_dir ../results_nl_sweep]
"""

import os
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Nonlinearity colour (Wong colour-blind-safe) and label
NL_ORDER  = ['kerr', 'saturable', 'tpa']
NL_COLOR  = {'kerr': '#0072B2', 'saturable': '#D55E00', 'tpa': '#009E73'}
NL_LABEL  = {'kerr': 'Kerr', 'saturable': 'Saturable', 'tpa': 'Two-photon abs.'}

# Global style: big fonts for a paper 
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


def lighten(color, frac=0.55):
    """Blend a colour toward white by `frac` (0 = unchanged, 1 = white)."""
    r, g, b = matplotlib.colors.to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def load(results_dir):
    """(nl, noise) -> stacked arrays: loss (S×E), phot (S×E)."""
    runs = defaultdict(lambda: {'loss': [], 'phot': []})
    for f in sorted(glob.glob(os.path.join(results_dir, '*.npz'))):
        d = np.load(f, allow_pickle=True)
        key = (str(d['nl']), float(d['noise_scale']))
        runs[key]['loss'].append(d['loss_history'])
        runs[key]['phot'].append(d['phot_history'])
    out = {}
    for k, v in runs.items():
        out[k] = {'loss': np.vstack(v['loss']), 'phot': np.vstack(v['phot'])}
    return out


def fig_loss(runs, noise, outstem):
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    handles = []
    for nl in NL_ORDER:
        d = runs.get((nl, noise))
        if d is None:
            continue
        L  = d['loss']
        ep = np.arange(1, L.shape[1] + 1)
        c  = NL_COLOR[nl]
        for row in L:                                   # faint individual seeds
            ax.semilogy(ep, row, color=c, lw=1.0, alpha=0.18, zorder=1)
        ax.semilogy(ep, np.nanmean(L, 0), color=c, lw=3.2, zorder=3)
        handles.append(Line2D([0], [0], color=c, lw=3.2, label=NL_LABEL[nl]))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim(0, L.shape[1])
    ax.legend(handles=handles, frameon=False, loc='best')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{outstem}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {outstem}.pdf (+ .png)')


def fig_photon(runs, outstem):
    """One figure, 3 columns (one per nonlinearity); dark=noiseless, light=noisy."""
    nls = [nl for nl in NL_ORDER if (nl, 0.0) in runs or (nl, 1.0) in runs]
    fig, axes = plt.subplots(1, len(nls), figsize=(6.0 * len(nls), 6.0),
                             squeeze=False)
    for col, nl in enumerate(nls):
        ax = axes[0][col]
        for noise, frac, name in [(0.0, 0.0, 'noiseless'), (1.0, 0.55, 'noisy')]:
            d = runs.get((nl, noise))
            if d is None:
                continue
            P  = d['phot']
            ep = np.arange(1, P.shape[1] + 1)
            ax.plot(ep, np.nanmean(P, 0), color=lighten(NL_COLOR[nl], frac),
                    lw=3.2, zorder=3 if noise == 0.0 else 2, label=name)
        ax.set_xlabel('Epoch')
        ax.set_xlim(0, P.shape[1])
        ax.legend(frameon=False, loc='best', title=NL_LABEL[nl],
                  title_fontsize=22)
        if col == 0:
            ax.set_ylabel(r'$\langle n \rangle_\mathrm{avg}$')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{outstem}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {outstem}.pdf (+ .png)')




def fig_cos_init(runs_dir, outstem):
    """Table replacement: initial cosine similarity between the SB gradient and
    the reference (noiseless: exact/IFT; noisy: FD-CRN), mean +/- std over
    seeds, colour = nonlinearity, dark = noiseless / light = noisy."""
    import collections
    tab = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(runs_dir, '*.npz'))):
        d = np.load(f, allow_pickle=True)
        tab[(str(d['nl']), float(d['noise_scale']))].append(float(d['cos_init']))

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    xpos = {nl: i for i, nl in enumerate(NL_ORDER)}
    for nl in NL_ORDER:
        for noise, frac, dx in [(0.0, 0.0, -0.13), (1.0, 0.55, +0.13)]:
            vals = np.array(tab.get((nl, noise), []))
            if vals.size == 0:
                continue
            c = lighten(NL_COLOR[nl], frac)
            x = xpos[nl] + dx
            ax.errorbar([x], [vals.mean()], yerr=[vals.std()], fmt='o',
                        ms=13, color=c, capsize=6, elinewidth=2.2,
                        capthick=2.2, zorder=3)
            ax.plot([x] * len(vals), vals, 'o', ms=5, color=c, alpha=0.45,
                    zorder=2)
    ax.set_xticks(range(len(NL_ORDER)))
    ax.set_xticklabels([NL_LABEL[nl].replace('Two-photon abs.',
                        'Two-photon\nabs.') for nl in NL_ORDER])
    ax.set_ylabel(r'$\cos(\nabla_\theta C,\ \nabla^{SB_y}_\theta C)$')
    ax.set_xlim(-0.5, len(NL_ORDER) - 0.5)
    ax.set_ylim(0.85, 1.005)
    from matplotlib.lines import Line2D as _L2
    ax.legend(handles=[
        _L2([0], [0], marker='o', ls='', ms=13, color='0.25',
            label='noiseless'),
        _L2([0], [0], marker='o', ls='', ms=13, color='0.7', label='noisy')],
        frameon=False, loc='lower right')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{outstem}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {outstem}.pdf (+ .png)')




def fig_final_loss(runs, outstem, n_last=25, T_m_noisy=200.0):
    """Companion of fig_cos_init: final training loss (mean of the last n_last
    epochs, per seed) per nonlinearity; dark = noiseless, light = noisy. Big
    marker = median over seeds, faint dots = individual seeds. Dashed line:
    predicted measurement floor 1/(4 T_m) of the record-based estimator."""
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    xpos = {nl: i for i, nl in enumerate(NL_ORDER)}
    for nl in NL_ORDER:
        for noise, frac, dx in [(0.0, 0.0, -0.13), (1.0, 0.55, +0.13)]:
            d = runs.get((nl, noise))
            if d is None:
                continue
            vals = np.array([row[-n_last:].mean() for row in d['loss']])
            c = lighten(NL_COLOR[nl], frac)
            x = xpos[nl] + dx
            ax.semilogy([x] * len(vals), vals, 'o', ms=6, color=c,
                        alpha=0.45, zorder=2)
            ax.semilogy([x], [np.median(vals)], 'o', ms=14, color=c, zorder=3)
    ax.axhline(1.0 / (4.0 * T_m_noisy), color='k', ls='--', lw=1.6, alpha=0.6)
    ax.text(-0.42, 1.0 / (4.0 * T_m_noisy) / 1.45,
            r'$1/(4 \bar{\kappa} \, T_m)$', ha='left', va='top',
            fontsize=20, color='0.25')
    ax.set_xticks(range(len(NL_ORDER)))
    ax.set_xticklabels([NL_LABEL[nl].replace('Two-photon abs.',
                        'Two-photon\nabs.') for nl in NL_ORDER])
    ax.set_ylabel('Final loss')
    ax.set_xlim(-0.5, len(NL_ORDER) - 0.5)
    from matplotlib.lines import Line2D as _L2
    ax.legend(handles=[
        _L2([0], [0], marker='o', ls='', ms=13, color='0.25',
            label='noiseless'),
        _L2([0], [0], marker='o', ls='', ms=13, color='0.7', label='noisy')],
        frameon=False, loc='center right')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{outstem}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {outstem}.pdf (+ .png)')


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--results_dir', default=os.path.join(
        here, 'Data', 'results_nl_sc'))
    args = ap.parse_args()

    runs = load(args.results_dir)
    if not runs:
        raise SystemExit(f'no npz in {args.results_dir}')

    out = os.path.join(here, 'Figures')
    os.makedirs(out, exist_ok=True)
    fig_cos_init(args.results_dir,
                 os.path.join(out, 'different_NN_fig_cos_init'))
    fig_final_loss(runs,
                   os.path.join(out, 'different_NN_fig_final_loss'))
    fig_loss(runs, 0.0, os.path.join(out, 'different_NN_fig_loss_noiseless'))
    fig_loss(runs, 1.0, os.path.join(out, 'different_NN_fig_loss_noisy'))
    fig_photon(runs,     os.path.join(out, 'different_NN_fig_photon'))


if __name__ == '__main__':
    main()
