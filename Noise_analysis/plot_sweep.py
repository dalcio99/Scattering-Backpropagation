"""
plot_sweep.py  —  plots for results_gradient_sweep/

File pattern: {mode}_ga0{ga0:.3g}_ratio{ratio:.6g}_noise{noise}_seed{seed}.npz
Sweep variable: ga0_target = g * a0_in_sq / kappa^2

Figures produced:
  fig_s_{mode}_1_angle_{noise}.pdf        Angle° vs ratio, colours = ga0_target
  fig_s_{mode}_2_cos_{noise}.pdf          Cosine vs ratio, colours = ga0_target
  fig_s_{mode}_3_cos_heatmap_{noise}.pdf  Heatmap ga0 * ratio → cosine similarity
  fig_s_{mode}_4_loss_heatmap_{noise}.pdf Heatmap ga0 * ratio → avg loss
  fig_s_{mode}_5_loss_scatter_{noise}.pdf Loss vs ratio, colours = ga0_target
  fig_s_{mode}_6_loss_history.pdf         Loss histories (best ratio per ga0)

Usage
-----
  python plot_sweep.py
  python plot_sweep.py --results_dir results_gradient_sweep --out_dir figures_sweep
"""

import os
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

NOISE_DIR = os.path.dirname(os.path.abspath(__file__))
_CMAP = plt.cm.winter


def _ga0_color(rank, n_total):
    return _CMAP(0.10 + 0.78 * rank / max(n_total - 1, 1))


def _ga0_legend_label(ga0, geff=np.nan):
    if np.isfinite(geff):
        return (rf'$g\langle|\bar{{a}}_\mathrm{{in}}|^2\rangle_\mathrm{{avg}}/\bar{{\kappa}}^2={ga0:.1f}$'
                rf'  ($g\langle n\rangle_\mathrm{{avg}}/\bar{{\kappa}}={geff:.3g}$)')
    return rf'$g\langle|\bar{{a}}_\mathrm{{in}}|^2\rangle_\mathrm{{avg}}/\bar{{\kappa}}^2={ga0:.1f}$'


def _ga0_color_legend(ax, ga0_list, summary, **extra_kw):
    """Colored-box table legend outside ax: box | ga0_target  (geff)."""
    n       = len(ga0_list)
    handles = []
    labels  = []
    for ci, ga0 in enumerate(ga0_list):
        geff, _ = summary.get(ga0, (np.nan, np.nan))
        handles.append(Patch(facecolor=_ga0_color(ci, n),
                             edgecolor='.35', linewidth=0.5))
        gs = f'{geff:.3g}' if np.isfinite(geff) else 'n/a'
        labels.append(f'{ga0:.1f}  ({gs})')
    title = (r'$g\langle|\bar{a}_\mathrm{in}|^2\rangle_\mathrm{avg}/\bar{\kappa}^2$'
             '\n'
             r'  ($g\langle n\rangle_\mathrm{avg}/\bar{\kappa}$)')
    kw = dict(loc='upper left', bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, framealpha=0.9,
              handlelength=1.5, handleheight=1.2, title=title)
    kw.update(extra_kw)
    ax.legend(handles, labels, **kw)

# ---------------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------------

def load_main(results_dir, mode, n_last=10):
    """Load sweep .npz files for one mode -> list of row dicts."""
    pattern = os.path.join(results_dir, f'{mode}_ga0*.npz')
    files   = sorted(glob.glob(pattern))
    rows    = []

    for fpath in files:
        try:
            arr = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f'  [skip] {os.path.basename(fpath)}: {e}')
            continue

        lh    = arr['loss_history']
        ph    = arr['phot_history']
        rh    = arr['ratio_history']
        g     = float(arr['g'])
        kappa = float(arr.get('kappa_val', 1.0))
        ga0   = float(arr['ga0_target'])

        n_last_eff   = min(n_last, len(lh))
        geff_end     = g * float(ph[-1]) / kappa
        phot_end     = float(ph[-1])

        rows.append({
            'ga0_target'  : ga0,
            'g'           : g,
            'a0_in_sq'    : float(arr.get('a0_in_sq', 90.0)),
            'ratio'       : float(arr['ratio_init']),
            'noise'       : float(arr['noise_scale']),
            'seed'        : int(arr['seed']),
            'angle_init'  : float(arr.get('angle_deg_init', np.nan)),
            'cos_init'    : float(arr.get('cos_init',       np.nan)),
            'loss_last'   : float(np.nanmean(lh[-n_last_eff:])),
            'loss_min'    : float(np.nanmin(lh)),
            'geff_end'    : geff_end,
            'phot_end'    : phot_end,
            'loss_history': lh,
        })

    print(f'  [{mode}] loaded {len(rows)} runs')
    return rows


def _per_ga0_summary(rows):
    """Return {ga0_target: (mean_geff_end, mean_phot_end)} averaged over seeds/ratios."""
    acc_geff = defaultdict(list)
    acc_phot = defaultdict(list)
    for r in rows:
        acc_geff[r['ga0_target']].append(r['geff_end'])
        acc_phot[r['ga0_target']].append(r['phot_end'])
    summary = {}
    for ga0 in acc_geff:
        summary[ga0] = (float(np.nanmean(acc_geff[ga0])),
                        float(np.nanmean(acc_phot[ga0])))
    return summary

# ---------------------------------------------------------------------------
#.  Plot 1/2: angle or cosine vs ratio, colour = ga0_target
# ---------------------------------------------------------------------------

def plot_angle_cos(rows, noise_val, mode, out_dir,
                   y_key='angle_init', y_label='Angle [°]', ylim=(0, 92),
                   fname_tag='1_angle', no_std=False, no_title=False):
    subset = [r for r in rows if abs(r['noise'] - noise_val) < 1e-6]
    if not subset:
        return

    ga0_list  = sorted({r['ga0_target'] for r in subset})
    n_ga0     = len(ga0_list)
    noise_str = 'noise=1' if noise_val > 0 else 'no noise'
    summary   = _per_ga0_summary(subset)

    fig, ax = plt.subplots(figsize=(10.0, 5.5))

    for ci, ga0 in enumerate(ga0_list):
        geff, _ = summary.get(ga0, (np.nan, np.nan))
        pts     = [r for r in subset if r['ga0_target'] == ga0]
        col     = _ga0_color(ci, n_ga0)

        ratio_map = defaultdict(list)
        for r in pts:
            ratio_map[r['ratio']].append(r[y_key])

        ratios = sorted(ratio_map.keys())
        x_m = np.array(ratios)
        y_m = np.array([np.nanmean(ratio_map[r]) for r in ratios])
        y_s = np.array([np.nanstd( ratio_map[r]) for r in ratios])

        ok = np.isfinite(x_m) & np.isfinite(y_m) & (x_m > 0)
        if not ok.any():
            continue

        ax.plot(x_m[ok], y_m[ok], 'o-', color=col, lw=1.8, ms=5,
                label=_ga0_legend_label(ga0, geff))
        if not no_std:
            lo = y_m[ok] - y_s[ok]
            hi = y_m[ok] + y_s[ok]
            if ylim is not None:
                lo = np.maximum(lo, ylim[0])
                hi = np.minimum(hi, ylim[1])
            ax.fill_between(x_m[ok], lo, hi, color=col, alpha=0.15)

    all_x = [r['ratio'] for r in subset if r['ratio'] > 0]
    if all_x:
        ax.set_xlim(min(all_x), max(all_x))

    ax.set_xscale('log')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(r'$|\delta a_{\mathrm{in}}|/|a^0_{\mathrm{in}}|$')
    ax.set_ylabel(y_label)
    if not no_title:
        ax.set_title(f'[{mode}, {noise_str}]  {y_label}  vs nudge ratio')
    _ga0_color_legend(ax, ga0_list, summary)
    ax.grid(True, which='both', alpha=0.22)
    fig.tight_layout()

    noise_tag = 'noise' if noise_val > 0 else 'no_noise'
    fname = os.path.join(out_dir, f'fig_s_{mode}_{fname_tag}_{noise_tag}.pdf')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {fname}')

# ---------------------------------------------------------------------------
# Plot 2b: noiseless vs noisy cosine similarity side-by-side 
# ---------------------------------------------------------------------------

def plot_cos_noiseless_vs_noisy(rows, mode, out_dir, no_std=False, no_title=False,
                                compact=False):
    """Left: cosine similarity (noise=0), Right: cosine similarity (noise>0).
    compact=True: 3/4-width axes, no grid lines, saved with _compact suffix."""
    noise_0 = sorted([v for v in {r['noise'] for r in rows} if v == 0.0])
    noise_1 = sorted([v for v in {r['noise'] for r in rows} if v > 0.0])
    if not noise_0 or not noise_1:
        return

    ga0_list = sorted({r['ga0_target'] for r in rows})
    n_ga0    = len(ga0_list)
    summary  = _per_ga0_summary(
        [r for r in rows if abs(r['noise'] - noise_1[0]) < 1e-6])

    fw = 13.5 if compact else 18.0
    fig, axes = plt.subplots(1, 2, figsize=(fw, 5.5), sharey=True)

    for ax, noise_val in zip(axes, [noise_0[0], noise_1[0]]):
        subset = [r for r in rows if abs(r['noise'] - noise_val) < 1e-6]

        for ci, ga0 in enumerate(ga0_list):
            pts      = [r for r in subset if r['ga0_target'] == ga0]
            col      = _ga0_color(ci, n_ga0)
            ratio_map = defaultdict(list)
            for r in pts:
                ratio_map[r['ratio']].append(r['cos_init'])

            ratios = sorted(ratio_map.keys())
            x_m = np.array(ratios)
            y_m = np.array([np.nanmean(ratio_map[r]) for r in ratios])
            y_s = np.array([np.nanstd( ratio_map[r]) for r in ratios])
            ok  = np.isfinite(x_m) & np.isfinite(y_m) & (x_m > 0)
            if not ok.any():
                continue

            ax.plot(x_m[ok], y_m[ok], 'o-', color=col, lw=1.8, ms=5)
            if not no_std:
                lo = np.maximum(y_m[ok] - y_s[ok], -0.1)
                hi = np.minimum(y_m[ok] + y_s[ok],  1.05)
                ax.fill_between(x_m[ok], lo, hi, color=col, alpha=0.15)

        all_x = [r['ratio'] for r in subset if r['ratio'] > 0]
        if all_x:
            ax.set_xlim(min(all_x), max(all_x))
        ax.set_xscale('log')
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlabel(r'$|\delta a_{\mathrm{in}}|/|a^0_{\mathrm{in}}|$')
        if not compact:
            ax.grid(True, which='both', alpha=0.22)
        if not no_title:
            label = 'Noiseless' if noise_val == 0.0 else 'Noisy'
            ax.set_title(f'[{mode}]  Cosine similarity  —  {label}')

    axes[0].set_ylabel('Cosine similarity SB vs true')
    _ga0_color_legend(axes[1], ga0_list, summary)
    fig.tight_layout()

    suffix = '_compact' if compact else ''
    fname = os.path.join(out_dir, f'fig_s_{mode}_cos_noise_comparison{suffix}.pdf')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {fname}')

# ---------------------------------------------------------------------------
#  Plot 3/4: heatmap  ga0_target * ratio 
# ---------------------------------------------------------------------------

def _log_bin_edges(vals):
    """Geometric-mean bin edges for an array of log-spaced values."""
    lv = np.log10(np.array(sorted(vals), dtype=float))
    edges = np.empty(len(lv) + 1)
    edges[1:-1] = (lv[:-1] + lv[1:]) / 2
    edges[0]  = lv[0]  - (lv[1]  - lv[0])  / 2
    edges[-1] = lv[-1] + (lv[-1] - lv[-2]) / 2
    return 10**edges


def plot_heatmap(rows, noise_val, mode, out_dir, value_key='cos_init',
                 cmap='RdYlGn', log_scale=False, fname_tag='3_cos_heatmap',
                 cbar_label='Cosine similarity SB vs true',
                 annotate=True, vmin=None, vmax=None, no_title=False):
    subset = [r for r in rows if abs(r['noise'] - noise_val) < 1e-6]
    if not subset:
        return

    ga0s   = sorted({r['ga0_target'] for r in subset})
    ratios = sorted({r['ratio'] for r in subset})
    nc, nr = len(ga0s), len(ratios)
    ga0_idx   = {g: i for i, g in enumerate(ga0s)}
    ratio_idx = {r: j for j, r in enumerate(ratios)}

    acc = defaultdict(list)
    for r in subset:
        key = (r['ga0_target'], r['ratio'])
        v = r.get(value_key, np.nan)
        if np.isfinite(float(v)):
            acc[key].append(float(v))

    grid = np.full((nc, nr), np.nan)
    for (ga0, ratio), vals in acc.items():
        grid[ga0_idx[ga0], ratio_idx[ratio]] = np.nanmean(vals)

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return

    v0 = vmin if vmin is not None else finite.min()
    v1 = vmax if vmax is not None else finite.max()
    norm = (mcolors.LogNorm(vmin=max(v0, 1e-9), vmax=v1)
            if log_scale else mcolors.Normalize(vmin=v0, vmax=v1))

    # True log x bin edges, uniform y index edges
    edges_x = _log_bin_edges(ratios)
    edges_y = np.arange(nc + 1, dtype=float) - 0.5
    X, Y   = np.meshgrid(edges_x, edges_y)

    # Per-ga0 summary for 3-axis y labels
    summary = _per_ga0_summary(subset)

    ax3_outward_pt  = 100
    right_margin_in = 3.2   # room for ax2, ax3, colorbar + all labels
    left_margin_in  = 0.90
    axes_width_in   = max(4.5, nr * 0.50)
    fig_w = left_margin_in + axes_width_in + right_margin_in
    fig_h = max(3.5, nc * 0.65 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    left_frac  = left_margin_in / fig_w
    right_frac = 1.0 - right_margin_in / fig_w
    fig.subplots_adjust(left=left_frac, right=right_frac, top=0.87, bottom=0.22)

    pcm = ax.pcolormesh(X, Y, grid, cmap=cmap, norm=norm, shading='flat')
    ax.set_xscale('log')

    if annotate:
        col_x = np.array([np.sqrt(edges_x[j] * edges_x[j+1]) for j in range(nr)])
        for i in range(nc):
            for j in range(nr):
                val = grid[i, j]
                if not np.isfinite(val):
                    ax.text(col_x[j], i, 'n/a', ha='center', va='center',
                            fontsize=7, color='gray')
                    continue
                nv = norm(val)
                fc = 'white' if (nv < 0.35 or nv > 0.78) else 'black'
                ax.text(col_x[j], i, f'{val:.2g}',
                        ha='center', va='center', fontsize=7.5, color=fc)

    ax.set_xticks(ratios)
    ax.set_xticklabels([f'{r:.2g}' for r in ratios], rotation=45)
    ax.set_xticks([], minor=True)
    ax.set_xlabel(r'$|\delta a_{\mathrm{in}}|/|a^0_{\mathrm{in}}|$')

    row_y = np.arange(nc, dtype=float)
    ax.set_yticks(row_y)
    ax.set_yticklabels([f'{g:.1f}' for g in ga0s])
    ax.set_ylabel(r'$g\langle|\bar{a}_\mathrm{in}|^2\rangle_\mathrm{avg}/\bar{\kappa}^2$')

    def _fmt_geff(v):
        if not np.isfinite(v):
            return 'n/a'
        return '0.0' if v == 0.0 else f'{v:.3g}'

    # Right axis 1: g·⟨n_j⟩/κ̄  (measured g_eff)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(row_y)
    ax2.set_yticklabels([_fmt_geff(summary.get(g, (np.nan,))[0]) for g in ga0s])
    ax2.set_ylabel(r'$g\langle n\rangle_\mathrm{avg}/\bar{\kappa}$')

    # Right axis 2: ⟨n_j⟩, offset further right
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', ax3_outward_pt))
    ax3.set_ylim(ax.get_ylim())
    ax3.set_yticks(row_y)
    ax3.set_yticklabels([f'{summary.get(g, (np.nan, np.nan))[1]:.1f}' for g in ga0s])
    ax3.set_ylabel(r'$\langle n\rangle_\mathrm{avg}$')

    noise_str = 'noise=1' if noise_val > 0 else 'no noise'
    if not no_title:
        ax.set_title(f'[{mode}, {noise_str}]  {cbar_label}', fontsize=11)

    pos = ax.get_position()
    # Colorbar placed after ax2 labels (~55pt) + ax3 spine (ax3_outward_pt) + ax3 labels (~60pt)
    clearance_frac = (55 + ax3_outward_pt + 65) / (72.0 * fig_w)
    cax  = fig.add_axes([pos.x1 + clearance_frac, pos.y0, 0.025, pos.height])
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label(cbar_label)

    noise_tag = 'noise' if noise_val > 0 else 'no_noise'
    fname = os.path.join(out_dir, f'fig_s_{mode}_{fname_tag}_{noise_tag}.pdf')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {fname}')

# ---------------------------------------------------------------------------
#   Plot 5: loss vs ratio, colour = ga0_target 
# ---------------------------------------------------------------------------


def plot_loss_scatter(rows, noise_val, mode, out_dir,
                      value_key='loss_min', ylabel=None, fname_tag='5_loss_scatter',
                      no_title=False):
    subset = [r for r in rows if abs(r['noise'] - noise_val) < 1e-6]
    if not subset:
        return

    ga0_list  = sorted({r['ga0_target'] for r in subset})
    n_ga0     = len(ga0_list)
    noise_str = 'noise=1' if noise_val > 0 else 'no noise'
    summary   = _per_ga0_summary(subset)
    if ylabel is None:
        ylabel = ('Min loss (over epochs, mean over seeds)' if value_key == 'loss_min'
                  else 'Avg loss (last 10 epochs, mean over seeds)')

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    for ci, ga0 in enumerate(ga0_list):
        geff, _ = summary.get(ga0, (np.nan, np.nan))
        pts     = [r for r in subset if r['ga0_target'] == ga0]
        col     = _ga0_color(ci, n_ga0)

        ratio_map = defaultdict(list)
        for r in pts:
            ratio_map[r['ratio']].append(r[value_key])

        ratios = sorted(ratio_map.keys())
        x_m = np.array(ratios)
        y_m = np.array([np.nanmean(ratio_map[r]) for r in ratios])
        y_s = np.array([np.nanstd( ratio_map[r]) for r in ratios])

        ok = np.isfinite(x_m) & np.isfinite(y_m) & (y_m > 0) & (x_m > 0)
        if not ok.any():
            continue

        ax.plot(x_m[ok], y_m[ok], 'o-', color=col, lw=1.8, ms=5,
                label=_ga0_legend_label(ga0, geff))
        lo = np.maximum(y_m[ok] - y_s[ok], 1e-9)
        ax.fill_between(x_m[ok], lo, y_m[ok] + y_s[ok], color=col, alpha=0.15)

    all_x = [r['ratio'] for r in subset if r['ratio'] > 0]
    if all_x:
        ax.set_xlim(min(all_x), max(all_x))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$|\delta a_{\mathrm{in}}|/|a^0_{\mathrm{in}}|$')
    ax.set_ylabel(ylabel)
    if not no_title:
        ax.set_title(f'[{mode}, {noise_str}]  loss vs perturbation ratio')
    _ga0_color_legend(ax, ga0_list, summary)
    ax.grid(True, which='both', alpha=0.22)
    fig.tight_layout()

    noise_tag = 'noise' if noise_val > 0 else 'no_noise'
    fname = os.path.join(out_dir, f'fig_s_{mode}_{fname_tag}_{noise_tag}.pdf')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {fname}')

# ---------------------------------------------------------------------------
#   Plot 6: loss histories vs epoch 
# ---------------------------------------------------------------------------


def plot_loss_histories(rows, mode, out_dir, ratio_target=0.038, no_title=False):
    if not rows:
        return

    noise_vals = sorted({r['noise'] for r in rows})
    ga0_list   = sorted({r['ga0_target'] for r in rows})
    n_ga0      = len(ga0_list)
    n_panels   = len(noise_vals)
    summary    = _per_ga0_summary(rows)

    fig, axes = plt.subplots(1, n_panels, figsize=(9.0 * n_panels, 5.5),
                             sharey=False, squeeze=False)

    for pi, noise_val in enumerate(noise_vals):
        ax        = axes[0, pi]
        noise_str = 'noise=1' if noise_val > 0 else 'no noise'
        subset    = [r for r in rows if abs(r['noise'] - noise_val) < 1e-6]

        for ci, ga0 in enumerate(ga0_list):
            geff, _ = summary.get(ga0, (np.nan, np.nan))
            col  = _ga0_color(ci, n_ga0)
            pts  = [r for r in subset
                    if r['ga0_target'] == ga0
                    and abs(r['ratio'] - ratio_target) / (ratio_target + 1e-12) < 0.10]
            if not pts:
                continue

            lhs = [r['loss_history'] for r in pts]
            nep = min(len(l) for l in lhs)
            mat = np.vstack([l[:nep] for l in lhs]).astype(float)
            avg = mat.mean(axis=0)
            ep  = np.arange(1, nep + 1)

            label = _ga0_legend_label(ga0, geff)
            ax.semilogy(ep, avg, lw=2.0, color=col, label=label)

        if no_title:                        # paper style: bigger fonts
            ax.set_xlabel('Epoch', fontsize=22)
            ax.set_ylabel('Loss', fontsize=22)
            ax.tick_params(labelsize=19)
        else:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'[{mode}, {noise_str}]  loss history\n'
                         rf'(ratio$\approx${ratio_target:.4g}, mean over seeds)')
            ax.grid(True, which='both', alpha=0.22)   # paper style: no grid

    if no_title:
        _ga0_color_legend(axes[0, -1], ga0_list, summary,
                          fontsize=16, title_fontsize=16)
    else:
        _ga0_color_legend(axes[0, -1], ga0_list, summary)
    fig.tight_layout()
    rtag  = f'{ratio_target:.4g}'.replace('.', 'p')
    fname = os.path.join(out_dir, f'fig_s_{mode}_6_loss_history_r{rtag}.pdf')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {fname}')

# ---------------------------------------------------------------------------
# Plot 7: photon number vs ga0_target, per seed 
# ---------------------------------------------------------------------------

def plot_phot_vs_ga0(rows_x, rows_y, out_dir, no_title=False):
    """
    ⟨n_end⟩ vs ga0_target for both modes, all seeds.
    One panel per mode. Each seed is a separate line (thin); mean over seeds is bold.
    Also shows ⟨n⟩ = a0_in_sq (linear-limit reference) as a dashed horizontal.
    """
    noise_vals = sorted({r['noise'] for r in rows_x + rows_y})

    for noise_val in noise_vals:
        noise_str = 'noise=1' if noise_val > 0 else 'no noise'
        noise_tag = 'noise' if noise_val > 0 else 'no_noise'

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        if not no_title:
            fig.suptitle(rf'Mean photon number $\langle n\rangle_\mathrm{{avg}}$ vs nonlinearity  [{noise_str}]',
                         fontsize=12)

        for ci, (rows, mode) in enumerate([(rows_x, 'sigma_x'), (rows_y, 'sigma_y')]):
            ax     = axes[ci]
            subset = [r for r in rows if abs(r['noise'] - noise_val) < 1e-6]
            if not subset:
                if not no_title:
                    ax.set_title(f'{mode}  (no data)', fontsize=11)
                continue

            ga0s  = sorted({r['ga0_target'] for r in subset})
            seeds = sorted({r['seed'] for r in subset})

            # Per-seed lines (averaged over ratios for each ga0)
            seed_curves = {}
            for seed in seeds:
                pts = [r for r in subset if r['seed'] == seed]
                y   = []
                for ga0 in ga0s:
                    vals = [r['phot_end'] for r in pts if r['ga0_target'] == ga0
                            and np.isfinite(r['phot_end'])]
                    y.append(float(np.nanmean(vals)) if vals else np.nan)
                seed_curves[seed] = np.array(y)
                ax.plot(ga0s, y, 'o--', lw=1.2, ms=5, alpha=0.6,
                        label=f'seed {seed}')

            # Mean over seeds
            mat  = np.vstack(list(seed_curves.values()))
            mean = np.nanmean(mat, axis=0)
            std  = np.nanstd(mat, axis=0) if mat.shape[0] > 1 else np.zeros(len(ga0s))
            ax.plot(ga0s, mean, 'o-', lw=2.2, ms=6, color='black', label='mean', zorder=5)
            ax.fill_between(ga0s,
                            np.maximum(mean - std, 0),
                            mean + std,
                            color='black', alpha=0.12)

            # Linear-limit reference: ⟨n⟩ ≈ a0_in_sq at g=0
            a0_in_sq_ref = float(subset[0]['a0_in_sq']) if subset else 90.0
            ax.axhline(a0_in_sq_ref, ls=':', color='gray', lw=1.3,
                       label=rf'$|\alpha_0|^2={a0_in_sq_ref:.0f}$ (linear limit)')

            ax.set_xlabel(r'$g\langle|\bar{a}_\mathrm{in}|^2\rangle_\mathrm{avg}/\bar{\kappa}^2$')
            if ci == 0:
                ax.set_ylabel(r'$\langle n\rangle_\mathrm{avg}$  (mean over ratios & patterns)')
            if not no_title:
                ax.set_title(f'{mode}')
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.22)
            ax.legend(loc='upper right', framealpha=0.9)

        fig.tight_layout()
        fname = os.path.join(out_dir, f'fig_s_7_phot_vs_ga0_{noise_tag}.pdf')
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved  {fname}')

# ---------------------------------------------------------------------------
# Plot: σ_x vs σ_y comparison
# ---------------------------------------------------------------------------

def plot_mode_comparison(rows_x, rows_y, out_dir, no_title=False):
    all_ga0s = sorted({r['ga0_target'] for r in rows_x + rows_y})
    n_ga0    = len(all_ga0s)

    noise_vals = sorted({r['noise'] for r in rows_x + rows_y})
    n_noise    = len(noise_vals)

    fig, axes = plt.subplots(n_noise, 2, figsize=(13, 4.5 * n_noise),
                             sharex=True, squeeze=False)
    if not no_title:
        fig.suptitle(r'Gradient alignment: $\sigma_x$ vs $\sigma_y$ SB', fontsize=12)

    for ri, noise_val in enumerate(noise_vals):
        noise_str = 'noise=1' if noise_val > 0 else 'no noise'
        for ci, (rows, mode) in enumerate([(rows_x, 'sigma_x'),
                                            (rows_y, 'sigma_y')]):
            ax     = axes[ri, ci]
            subset = [r for r in rows if abs(r['noise'] - noise_val) < 1e-6]
            summary = _per_ga0_summary(subset)

            for gi, ga0 in enumerate(all_ga0s):
                col = _ga0_color(gi, n_ga0)
                pts = [r for r in subset if r['ga0_target'] == ga0]
                if not pts:
                    continue
                ratio_map = defaultdict(list)
                for r in pts:
                    ratio_map[r['ratio']].append(r['angle_init'])
                ratios = sorted(ratio_map.keys())
                x_m = np.array(ratios)
                y_m = np.array([np.nanmean(ratio_map[r]) for r in ratios])
                ok  = np.isfinite(x_m) & np.isfinite(y_m) & (x_m > 0)
                if ok.any():
                    geff = summary.get(ga0, (np.nan,))[0]
                    ax.plot(x_m[ok], y_m[ok], 'o-', color=col, lw=1.6, ms=4,
                            label=_ga0_legend_label(ga0, geff))

            ax.set_xscale('log')
            all_x = [r['ratio'] for r in subset if r['ratio'] > 0]
            if all_x:
                ax.set_xlim(min(all_x) / 2.5, max(all_x) * 2.5)
            ax.set_ylim(0, 92)
            ax.grid(True, which='both', alpha=0.22)
            if not no_title:
                ax.set_title(f'{mode},  {noise_str}', fontsize=10)
            if ri == n_noise - 1:
                ax.set_xlabel(r'$|\delta a_{\mathrm{in}}|/|a^0_{\mathrm{in}}|$')
            if ci == 0:
                ax.set_ylabel('Angle SB vs true grad  [°]')
            if ri == 0 and ci == 1:
                _ga0_color_legend(ax, all_ga0s, _per_ga0_summary(
                    [r for r in rows_x + rows_y
                     if abs(r['noise'] - noise_val) < 1e-6]))

    fig.tight_layout()
    fname = os.path.join(out_dir, 'fig_s_sx_sy_comparison.pdf')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {fname}')

# ---------------------------------------------------------------------------
#  Main 
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', type=str,
                    default=os.path.join(NOISE_DIR, 'Data', 'results_sc_sweep20'))
    ap.add_argument('--out_dir',    type=str, default=None)
    ap.add_argument('--n_last',     type=int, default=10,
                    help='Window size for loss_last metric (default: 10)')
    ap.add_argument('--noise_only', action='store_true',
                    help='Only produce plots for the noisy condition (noise=1)')
    ap.add_argument('--fontsize',   type=int, default=11,
                    help='Base font size (default: 11, paper: 14)')
    ap.add_argument('--no_std',     action='store_true',
                    help='Omit std-dev shading in angle/cosine plots')
    ap.add_argument('--no_title',   action='store_true',
                    help='Omit all figure and axes titles')
    args = ap.parse_args()

    plt.rcParams.update({
        'font.size'      : args.fontsize,
        'axes.labelsize' : args.fontsize,
        'axes.titlesize' : args.fontsize,
        'xtick.labelsize': max(args.fontsize - 2, 8),
        'ytick.labelsize': max(args.fontsize - 2, 8),
        'legend.fontsize': max(args.fontsize - 3, 7),
    })

    out_dir = args.out_dir or os.path.join(
        NOISE_DIR, 'Figures', os.path.basename(os.path.normpath(args.results_dir)))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    last_label = f'Avg loss last {args.n_last} epochs (mean over seeds)'

    all_rows = {}
    for mode in ['sigma_x', 'sigma_y']:
        print(f'\n── {mode} ──')
        rows = load_main(args.results_dir, mode, n_last=args.n_last)
        all_rows[mode] = rows

        if not rows:
            print(f'  No data for {mode}.')
            continue

        noise_vals = sorted({r['noise'] for r in rows})
        if args.noise_only:
            noise_vals = [v for v in noise_vals if v > 0]

        for noise_val in noise_vals:
            plot_angle_cos(rows, noise_val, mode, out_dir,
                           y_key='angle_init',
                           y_label='Gradient angle SB vs true  [°]',
                           ylim=(0, 92), fname_tag='1_angle',
                           no_std=args.no_std, no_title=args.no_title)
            plot_angle_cos(rows, noise_val, mode, out_dir,
                           y_key='cos_init',
                           y_label='Cosine similarity SB vs true',
                           ylim=(-0.1, 1.05), fname_tag='2_cos',
                           no_std=args.no_std, no_title=args.no_title)
            plot_heatmap(rows, noise_val, mode, out_dir,
                         value_key='cos_init', cmap='RdYlGn',
                         cbar_label='Cosine similarity SB vs true',
                         vmin=-1, vmax=1, fname_tag='3_cos_heatmap',
                         no_title=args.no_title)
            plot_heatmap(rows, noise_val, mode, out_dir,
                         value_key='loss_min', cmap='plasma_r', log_scale=True,
                         cbar_label='Min loss (mean over seeds)',
                         fname_tag='4_loss_heatmap', no_title=args.no_title)
            plot_heatmap(rows, noise_val, mode, out_dir,
                         value_key='loss_last', cmap='plasma_r', log_scale=True,
                         cbar_label=last_label,
                         fname_tag='4b_loss_heatmap_last10', no_title=args.no_title)
            plot_loss_scatter(rows, noise_val, mode, out_dir,
                              no_title=args.no_title)
            plot_loss_scatter(rows, noise_val, mode, out_dir,
                              value_key='loss_last', fname_tag='5b_loss_scatter_last10',
                              no_title=args.no_title)

        for ratio in [0.000910716, 0.00190186, 0.00397166, 0.00829404,
                      0.0173205, 0.0361705, 0.0755352, 0.157741,
                      0.329411]:
            plot_loss_histories(rows, mode, out_dir, ratio_target=ratio,
                                no_title=args.no_title)

        plot_cos_noiseless_vs_noisy(rows, mode, out_dir,
                                    no_std=args.no_std, no_title=args.no_title)
        plot_cos_noiseless_vs_noisy(rows, mode, out_dir,
                                    no_std=args.no_std, no_title=args.no_title,
                                    compact=True)

    if all_rows.get('sigma_x') and all_rows.get('sigma_y'):
        plot_mode_comparison(all_rows['sigma_x'], all_rows['sigma_y'], out_dir,
                             no_title=args.no_title)
        plot_phot_vs_ga0(all_rows['sigma_x'], all_rows['sigma_y'], out_dir,
                         no_title=args.no_title)

    print('\nDone.')


if __name__ == '__main__':
    main()
