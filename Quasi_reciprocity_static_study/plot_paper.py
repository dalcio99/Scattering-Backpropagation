"""
plot_paper.py
PDF versions of four figures from ../angle_vs_params_dual:
    1. scatter_cosine_faded
    2. heatmap_convergence
    3. heatmap_Gz_cosine
    4. dist_cosGz_vs_g

Run:  /opt/miniconda3/envs/jax-env/bin/python plot_paper.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, 'Data')   # release package: data ships in ./Data
OUT  = os.path.join(HERE, 'Figures'); os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# global style (large fonts for print)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size'        : 24,
    'axes.titlesize'   : 24,
    'axes.labelsize'   : 26,
    'xtick.labelsize'  : 22,
    'ytick.labelsize'  : 22,
    'legend.fontsize'  : 22,
    'axes.linewidth'   : 1.5,
    'xtick.major.size' : 7,
    'ytick.major.size' : 7,
    'pdf.fonttype'     : 42,   # editable text in the PDF
    'ps.fonttype'      : 42,
})

# ---------------------------------------------------------------------------
# load 
# ---------------------------------------------------------------------------
d = np.load(os.path.join(SRC, 'angle_stats_dual.npz'))
mean_cosSx = d['mean_cosSx'];  std_cosSx  = d['std_cosSx']
mean_cosSz = d['mean_cosSz'];  std_cosSz  = d['std_cosSz']
mean_cosGx = d['mean_cosGx'];  std_cosGx  = d['std_cosGx']
mean_cosGz = d['mean_cosGz'];  std_cosGz  = d['std_cosGz']
raw_cosGz = d['raw_cosGz']
mean_geff = d['mean_geff']
succ_frac = d['succ_frac']
N_values = d['N_values']
g_values = d['g_values']

nC = 6;  nN = len(N_values);  ng = len(g_values)

conditions = [
    ("Self-Kerr",               r"$\sigma^2=1$"),
    ("Self-Kerr",               r"$\sigma^2=1/N$"),
    ("Cross-Kerr (n.n.)",       r"$\sigma^2=1$"),
    ("Cross-Kerr (n.n.)",       r"$\sigma^2=1/N$"),
    ("Cross-Kerr (all-to-all)", r"$\sigma^2=1$"),
    ("Cross-Kerr (all-to-all)", r"$\sigma^2=1/N$"),
]

# a-basis metric labels (code sig_x -> sigma_y ; code sig_z -> sigma_x)
LCOS_SY = r"$\cos(S^\dagger,\;\sigma_y S \sigma_y)$"
LCOS_SX = r"$\cos(S^\dagger,\;\sigma_x S \sigma_x)$"
LCOS_GY = r"$\cos(\nabla_\theta C,\;\nabla^{\rm SBy}_\theta C)$"
LCOS_GX = r"$\cos(\nabla_\theta C,\;\nabla^{\rm SBx}_\theta C)$"

#requested new axis labels
XLAB_G    = r"$g\,\langle |a_{\mathrm{in}}|^2 \rangle_{\mathrm{avg}}/\bar{\kappa}^2$"
XLAB_GEFF = r"$g\,\langle |\bar{a}|^2 \rangle_{\mathrm{avg}}/\bar{\kappa}$"

COS_CMAP  = 'RdBu'
CONV_CMAP = 'RdYlGn'

cmap_N = plt.cm.plasma_r
norm_N = mcolors.Normalize(vmin=N_values.min(), vmax=N_values.max())

g_ticks    = np.array([0, 0.5, 1.0, 2, 4, 6, 8])
g_tick_pos = [np.argmin(np.abs(g_values - v)) for v in g_ticks]

def _apply_gaxis(ax):
    ax.set_xticks(g_tick_pos)
    # ":g" drops the trailing ".0" on whole numbers (1.0 -> 1) while keeping 0.5
    ax.set_xticklabels([f"{v:g}" for v in g_ticks])
    ax.set_xlabel(XLAB_G)

# ---------------------------------------------------------------------------
#  1. scatter_cosine_faded
# ---------------------------------------------------------------------------
def _draw_faded(ax, gs, vals, stds, conv, color, std_scale, lw=1.9):
    alpha_pts = np.where(np.isfinite(conv), 0.15 + 0.75 * conv, 0.0)
    for gi in range(len(gs) - 1):
        v0, v1 = vals[gi], vals[gi+1]
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        a = float((alpha_pts[gi] + alpha_pts[gi+1]) / 2)
        ax.plot([gs[gi], gs[gi+1]], [v0, v1], '-', color=color, alpha=a, lw=lw)
    for gi in range(len(gs)):
        if not np.isfinite(vals[gi]):
            continue
        s = stds[gi]
        sz = 16 + 100*float(min(max(s, 0), std_scale)/std_scale) if np.isfinite(s) else 16
        ax.scatter(gs[gi], vals[gi], color=color, s=sz,
                   alpha=float(alpha_pts[gi]), zorder=3, linewidths=0)

def _add_geff_axis(ax, ci):
    """Secondary top x-axis with g_eff labels; ticks at 0.5 and 1.0 removed."""
    geff_vs_gi = np.nanmean(mean_geff[ci], axis=0)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_pos, tick_labels = [], []
    for tg in [0, 2, 4, 6, 8]:           # 0.5 and 1.0 removed on request
        idx = np.argmin(np.abs(g_values - tg))
        gv  = geff_vs_gi[idx]
        tick_pos.append(tg)
        tick_labels.append(f"{gv:.2g}" if np.isfinite(gv) else "")
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels)
    ax2.tick_params(axis='x', pad=8)          # space between axis and its values
    ax2.set_xlabel(XLAB_GEFF, labelpad=14)    # space between values and label

def scatter_cosine_faded():
    metrics = [
        (mean_cosSx, std_cosSx, LCOS_SY, 0.3),
        (mean_cosSz, std_cosSz, LCOS_SX, 0.3),
        (mean_cosGx, std_cosGx, LCOS_GY, 0.4),
        (mean_cosGz, std_cosGz, LCOS_GX, 0.4),
    ]
    nM   = len(metrics)
    cmap = plt.cm.copper          # colormap for N (requested)
    norm = norm_N
    # extra-large text for this figure only
    big = {'font.size': 28, 'axes.titlesize': 30, 'axes.labelsize': 34,
           'xtick.labelsize': 28, 'ytick.labelsize': 28, 'legend.fontsize': 28}
    with plt.rc_context(big):
        fig, axes = plt.subplots(nM, nC, figsize=(5.8*nC, 4.9*nM), sharey='row',
                                 constrained_layout=True)
        # interior rows sit close; the bottom row carries the x-axis label + ticks
        fig.set_constrained_layout_pads(hspace=0.02, h_pad=0.02)
        for mi, (mdata, sdata, ylabel, std_scale) in enumerate(metrics):
            is_bottom = (mi == nM - 1)
            for ci, (lk, lj) in enumerate(conditions):
                ax = axes[mi, ci]
                for ni, N in enumerate(N_values):
                    color = cmap(norm(N))
                    vals  = mdata[ci, ni, :]
                    stds  = sdata[ci, ni, :]
                    conv  = succ_frac[ci, ni, :]
                    if not np.any(np.isfinite(vals)):
                        continue
                    _draw_faded(ax, g_values, vals, stds, conv, color, std_scale)
                # dashed vertical line at g_eff = 1.0 removed on request
                ax.set_ylim(-1.05, 1.05); ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax.set_xticks([0, 2, 4, 6, 8])
                ax.grid(alpha=0.25)
                if is_bottom:
                    # bottom row shows the g|a_in|^2 axis (label + 0..8 values)
                    ax.set_xticklabels([str(v) for v in [0, 2, 4, 6, 8]])
                    ax.set_xlabel(XLAB_G)
                else:
                    ax.set_xticklabels([])          # interior rows: no numbers
                if ci == 0:
                    ax.set_ylabel(ylabel)
                if mi == 0:
                    # top row carries the g_eff axis + subtitle
                    _add_geff_axis(ax, ci)
                    ax.set_title(f"{lk}\n{lj}", pad=58)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, label=r"$N$", fraction=0.012, pad=0.02)
        cbar.set_ticks(N_values)
    # no suptitle
    fig.savefig(os.path.join(OUT, "scatter_cosine_faded.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Saved  scatter_cosine_faded.pdf")

# ---------------------------------------------------------------------------
#  heatmaps (shared colorbar)
# ---------------------------------------------------------------------------
def heatmap_figure(data, cmap, vmin, vmax, cbar_label=None):
    fig, axes = plt.subplots(3, 2, figsize=(16, 18), constrained_layout=True)
    # more vertical space between the rows of panels
    fig.set_constrained_layout_pads(h_pad=0.12, hspace=0.14)
    last_im = None
    for ci, (lk, lj) in enumerate(conditions):
        row, col = ci // 2, ci % 2
        ax = axes[row, col]
        im = ax.imshow(data[ci], cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', origin='lower',
                       extent=[-0.5, ng - 0.5, -0.5, nN - 0.5])
        last_im = im
        _apply_gaxis(ax)
        ax.set_yticks(np.arange(nN))
        ax.set_yticklabels(N_values)
        ax.set_ylabel(r"$N$")
        ax.set_title(f"{lk}   {lj}")
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    return fig

def save_heatmap(data, fname, cmap, vmin, vmax, cbar_label=None):
    big = {'font.size': 26, 'axes.titlesize': 28, 'axes.labelsize': 32,
           'xtick.labelsize': 24, 'ytick.labelsize': 24}
    with plt.rc_context(big):
        fig = heatmap_figure(data, cmap, vmin, vmax, cbar_label=cbar_label)
        fig.savefig(os.path.join(OUT, fname), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved  {fname}")

# ---------------------------------------------------------------------------
#  dist_cosGz_vs_g (violins)
# ---------------------------------------------------------------------------
G_SUB_IDX  = np.unique(np.round(np.linspace(0, ng-1, 10)).astype(int))
G_SUB_VALS = g_values[G_SUB_IDX]
N_SEL_VALS = np.array([20, 30, 40])
N_SEL_IDX  = [int(np.where(N_values == v)[0][0]) for v in N_SEL_VALS]
N_SEL_COLS = ['tab:red', 'tab:green', 'tab:blue']

def _add_violins(ax, vdata, vpos, vcols, valphas):
    if not vdata:
        return
    try:
        parts = ax.violinplot(vdata, positions=vpos,
                              widths=0.72, showmedians=True, showextrema=False)
        for body, col, alpha in zip(parts['bodies'], vcols, valphas):
            body.set_facecolor(col)
            body.set_alpha(float(np.clip(alpha, 0.05, 1.0)))
            body.set_edgecolor('none')
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(2.0)
    except Exception:
        for dat, p, c, a in zip(vdata, vpos, vcols, valphas):
            ax.scatter([p]*len(dat), dat, color=c,
                       alpha=float(np.clip(a, 0.05, 1.0))*0.6, s=8, linewidths=0)

def dist_cosGz_vs_g():
    raw_data = raw_cosGz
    ylabel   = r"$\cos(\nabla_\theta C,\;\nabla^{\rm SBx}_\theta C)$"
    ylim     = (-1.05, 1.05)
    n_sel    = len(N_SEL_IDX)
    spacing  = n_sel + 0.8
    big = {'font.size': 26, 'axes.titlesize': 28, 'axes.labelsize': 32,
           'xtick.labelsize': 22, 'ytick.labelsize': 24, 'legend.fontsize': 26}
    with plt.rc_context(big):
      fig, axes = plt.subplots(3, 2, figsize=(24, 26), constrained_layout=True)
      # more breathing room between panels (columns and rows); taller boxes
      fig.set_constrained_layout_pads(w_pad=0.15, h_pad=0.12,
                                      wspace=0.14, hspace=0.10)
      for ci, (lk, lj) in enumerate(conditions):
        row, col = ci // 2, ci % 2
        ax = axes[row, col]
        vdata, vpos, vcols, valphas = [], [], [], []
        group_ticks, g_labels, geff_labels = [], [], []
        for g_enum, gi in enumerate(G_SUB_IDX):
            base = g_enum * spacing
            group_ticks.append(base + (n_sel - 1) / 2.0)
            g_labels.append(f"{G_SUB_VALS[g_enum]:.1f}")
            gm = np.nanmean(mean_geff[ci, :, gi])
            geff_labels.append(f"{gm:.2g}" if np.isfinite(gm) else "")
            for k, ni in enumerate(N_SEL_IDX):
                s = raw_data[ci, ni, gi, :]
                s = s[np.isfinite(s)]
                if len(s) < 2:
                    continue
                alpha = 0.15 + 0.70 * float(succ_frac[ci, ni, gi])
                vdata.append(s); vpos.append(base + k)
                vcols.append(N_SEL_COLS[k]); valphas.append(alpha)
        _add_violins(ax, vdata, vpos, vcols, valphas)
        ax.set_xticks(group_ticks); ax.set_xticklabels(g_labels)
        ax.set_xlabel(XLAB_G)
        ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
        ax.grid(axis='y', alpha=0.25)
        ax.set_title(f"{lk}   {lj}", pad=62)   # space above the g_eff axis label
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(group_ticks)
        ax2.set_xticklabels(geff_labels)
        ax2.tick_params(axis='x', pad=8)
        ax2.set_xlabel(XLAB_GEFF, labelpad=18)   # more space between label and values
        if ci == 5:   # Cross-Kerr (all-to-all), sigma^2 = 1/N — most white space
            ax.legend(
                handles=[Patch(facecolor=N_SEL_COLS[k], alpha=0.75,
                               label=f"$N={N_SEL_VALS[k]}$")
                         for k in range(n_sel)],
                loc='center right')
      fig.savefig(os.path.join(OUT, "dist_cosGz_vs_g.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Saved  dist_cosGz_vs_g.pdf")

# ══════════════════════════════════════════════════════════════════════════════
#  single-panel: cos(S, sigma_y S† sigma_y), Self-Kerr, sigma^2 = 1/N
# ══════════════════════════════════════════════════════════════════════════════
def single_cosSy_selfkerr():
    ci        = 1          # Self-Kerr, sigma^2 = 1/N
    mdata     = mean_cosSx
    sdata     = std_cosSx
    std_scale = 0.3
    cmap      = plt.cm.copper
    norm      = mcolors.Normalize(vmin=N_values.min(), vmax=N_values.max())

    fig, ax = plt.subplots(figsize=(9.5, 8.0), constrained_layout=True)
    for ni, N in enumerate(N_values):
        color = cmap(norm(N))
        vals  = mdata[ci, ni, :]
        stds  = sdata[ci, ni, :]
        conv  = succ_frac[ci, ni, :]
        if not np.any(np.isfinite(vals)):
            continue
        _draw_faded(ax, g_values, vals, stds, conv, color, std_scale)
    ax.set_ylim(-0.02, 1.02); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xticklabels([str(v) for v in [0, 2, 4, 6, 8]])
    ax.grid(alpha=0.25)
    ax.set_xlabel(XLAB_G)                       # bottom: g <|a_in|^2>_avg / kbar^2
    ax.set_ylabel(LCOS_SY)

    _add_geff_axis(ax, ci)                       # top: g <|abar|^2>_avg / kbar
    ax.set_title(r"Self-Kerr   $\sigma^2=1/N$", pad=58)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=r"$N$", fraction=0.046, pad=0.02)
    cbar.set_ticks(N_values)

    fig.savefig(os.path.join(OUT, "cos_Sy_selfkerr_1overN.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Saved  cos_Sy_selfkerr_1overN.pdf")

# ---------------------------------------------------------------------------
#  convergence vs g, scatter style (1 x 6, one line per N)
# ---------------------------------------------------------------------------
def convergence_scatter():
    cmap = plt.cm.copper
    norm = norm_N
    big = {'font.size': 28, 'axes.titlesize': 30, 'axes.labelsize': 34,
           'xtick.labelsize': 28, 'ytick.labelsize': 28, 'legend.fontsize': 28}
    with plt.rc_context(big):
        fig, axes = plt.subplots(1, nC, figsize=(5.8*nC, 8.0), sharey=True,
                                 constrained_layout=True)
        for ci, (lk, lj) in enumerate(conditions):
            ax = axes[ci]
            for ni, N in enumerate(N_values):
                vals = succ_frac[ci, ni, :]
                fin  = np.isfinite(vals)
                if fin.sum() == 0:
                    continue
                ax.plot(g_values[fin], vals[fin], '-o', color=cmap(norm(N)),
                        lw=1.9, ms=5, alpha=0.9)
            ax.set_ylim(-0.02, 1.02); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticks([0, 2, 4, 6, 8])
            ax.set_xticklabels([str(v) for v in [0, 2, 4, 6, 8]])
            ax.grid(alpha=0.25)
            ax.set_xlabel(XLAB_G)                       # bottom: g <|a_in|^2>/kbar^2
            if ci == 0:
                ax.set_ylabel("convergence fraction")
            _add_geff_axis(ax, ci)                       # top: g <|abar|^2>/kbar
            ax.set_title(f"{lk}\n{lj}", pad=58)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, label=r"$N$", fraction=0.012, pad=0.02)
        cbar.set_ticks(N_values)
    fig.savefig(os.path.join(OUT, "convergence_scatter.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Saved  convergence_scatter.pdf")

# ── run ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scatter_cosine_faded()
    single_cosSy_selfkerr()
    convergence_scatter()
    save_heatmap(succ_frac, "heatmap_convergence.pdf", CONV_CMAP, 0, 1,
                 cbar_label="Convergence to steady state rate")
    save_heatmap(succ_frac,  "heatmap_convergence.pdf", CONV_CMAP, 0, 1,
                 cbar_label="Convergence to steady state rate")
    save_heatmap(mean_cosGz, "heatmap_Gz_cosine.pdf",   COS_CMAP, -1, 1,
                 cbar_label=LCOS_GX)
    dist_cosGz_vs_g()
    print("\nAll PDFs written to:", OUT)
