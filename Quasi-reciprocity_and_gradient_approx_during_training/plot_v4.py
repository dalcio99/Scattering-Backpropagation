"""
plot_v4.py  —  paper-ready figures from a training_angles_v4 results directory.

Reads <results_dir>/training_angles_v4.npz and regenerates the two main figures
(no retraining).  Usage:

    python plot_v4.py [results_dir]        # default: results_v4

Produces, in <results_dir>/:
    training_v4_fig1_sigmax.{pdf,png}
    training_v4_fig1_sigmay.{pdf,png}
    training_v4_fig2_method.{pdf,png}

Styling: large fonts, vector PDF, no 1e-4/1e-6 reference lines,
legend inside the S-quasi-symmetry (middle) row, plain-C gradient notation,
and the legend lists only g/κ̄ and g⟨|ā|²⟩/κ̄ (χ_in = g/κ̄ is redundant, dropped).
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size':        21,
    'axes.titlesize':   25,
    'axes.labelsize':   25,
    'xtick.labelsize':  19,
    'ytick.labelsize':  19,
    'legend.fontsize':  18,
    'figure.titlesize': 24,
    'pdf.fonttype':     42,     # editable text in the PDF
    'ps.fonttype':      42,
})

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else 'results_v4'
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = RESULTS_DIR if os.path.isabs(RESULTS_DIR) else os.path.join(_HERE, RESULTS_DIR)

d = np.load(os.path.join(OUT, 'training_angles_v4.npz'), allow_pickle=True)
loss    = d['loss']
cosSx   = d['cosSx'];  cosSy = d['cosSy']
cosGz   = d['cosGz'];  cosGx = d['cosGx']    # cosGz = SB_x, cosGx = SB_y
chi_ss  = d['chi_ss']
cmp_loss = d['cmp_loss']
g_values = list(d['g_values'])
kbar     = float(d['kbar'])
nK, nG, num_seeds, num_epochs = loss.shape

# column order matches the training script; label n.n. as line if this is an nnline run
_nn_label = "Cross-Kerr (n.n., line)" if 'nnline' in os.path.basename(OUT) else "Cross-Kerr (n.n.)"
kerr_names = ["Self-Kerr", _nn_label, "Cross-Kerr (all-to-all)"]

epochs   = np.arange(1, num_epochs + 1)
g_colors = plt.cm.viridis(np.linspace(0.05, 0.92, nG))

# χ_ss at the initial network (epoch 0), mean over seeds
chi_ss_init = np.nanmean(chi_ss[:, :, :, 0], axis=2)   # (nK, nG)

# χ_in = g⟨|a_in|²⟩/κ̄²  =  g/κ̄  exactly (input scaling gives ⟨|a_in|²⟩_avg = 1)
chi_in = np.array([g/kbar for g in g_values])          # (nG,)  same for every column

CHI_IN_EXPR = r"$g\langle|a_{\rm in}|^2\rangle_{\rm avg}/\bar\kappa^2$"
CHI_SS_EXPR = r"$g\langle|\bar a|^2\rangle_{\rm avg}/\bar\kappa$"
LEGEND_TITLE = CHI_IN_EXPR + "    " + CHI_SS_EXPR

def legend_handles_labels(ki):
    """Per-column legend: two aligned columns (χ_in, χ_ss) in x.xx format."""
    handles = [Line2D([0], [0], color=g_colors[gi], lw=2.6) for gi in range(nG)]
    labels  = [f"{chi_in[gi]:>6.2f}     {chi_ss_init[ki,gi]:>6.2f}" for gi in range(nG)]
    return handles, labels

def _save(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), bbox_inches='tight',
                    dpi=(200 if ext == 'png' else None))
    plt.close(fig)
    print(f"Saved  {stem}.pdf / .png")


# -------------------------------------------------------------------------
# FIGURE 1 (one per symmetry)
# --------------------------------------------------------------------------
def make_fig1(symm):
    if symm == 'x':
        cosS, cosG = cosSx, cosGz
        s_lab = r'$\cos(S^\dagger,\ \sigma_x S\,\sigma_x^{-1})$'
        g_lab = r'$\cos(\nabla_\theta C,\ \nabla_\theta^{{\rm SB}_x} C)$'
        title = r'$\sigma_x$ feedback (x-quadrature perturbation)'
    else:
        cosS, cosG = cosSy, cosGx
        s_lab = r'$\cos(S^\dagger,\ \sigma_y S\,\sigma_y^{-1})$'
        g_lab = r'$\cos(\nabla_\theta C,\ \nabla_\theta^{{\rm SB}_y} C)$'
        title = r'$\sigma_y$ feedback (p-quadrature perturbation)'

    rows = [(loss, 'MSE', True, None, None),
            (cosS, s_lab, False, -1.05, 1.05),
            (cosG, g_lab, False, -1.05, 1.05)]

    LEGEND_ROW = 1     # middle row (S quasi-symmetry) — one legend per column

    fig, axes = plt.subplots(3, nK, figsize=(6.2*nK, 4.0*3), sharey='row')
    for mi, (dat4d, mlabel, do_log, ymin, ymax) in enumerate(rows):
        for ki in range(nK):
            ax = axes[mi, ki]
            for gi, g in enumerate(g_values):
                mean = np.nanmean(dat4d[ki, gi], axis=0)
                if do_log:
                    for si in range(num_seeds):
                        ax.plot(epochs, dat4d[ki,gi,si], color=g_colors[gi], lw=0.5, alpha=0.18)
                    ax.plot(epochs, mean, color=g_colors[gi], lw=2.4)
                else:
                    std = np.nanstd(dat4d[ki, gi], axis=0)
                    ax.plot(epochs, mean, color=g_colors[gi], lw=2.4)
                    ax.fill_between(epochs, mean-std, mean+std, color=g_colors[gi], alpha=0.12)
            ax.set_xscale('log'); ax.set_xlim(epochs[0], epochs[-1]); ax.grid(alpha=0.25)
            if do_log:
                ax.set_yscale('log')
                ax.set_title(kerr_names[ki], fontsize=25)
            else:
                ax.set_ylim(ymin, ymax)
                ax.axhline(1.0, color='k', lw=0.8, ls=':', alpha=0.5)
            if mi == LEGEND_ROW:
                h, l = legend_handles_labels(ki)
                ax.legend(h, l, loc='lower center', title=LEGEND_TITLE, fontsize=13,
                          title_fontsize=13, labelspacing=0.28, handlelength=1.2,
                          borderpad=0.5, framealpha=0.9)
            if mi == 2: ax.set_xlabel("Epoch")
            if ki == 0: ax.set_ylabel(mlabel)
    fig.tight_layout()
    _save(fig, f"training_v4_fig1_sigma{symm}")

make_fig1('x')
make_fig1('y')

# ---------------------------------------------------------------------------
# FIGURE 2 — method comparison by nonlinearity
# ---------------------------------------------------------------------------
# Skipped when the npz carries no method-comparison data (all-NaN cmp_loss),
# e.g. a training_angles_v4_sigmay.py run with SB_RUN_CMP unset.
if np.all(np.isnan(cmp_loss)):
    print("cmp_loss is all-NaN (method comparison not run) — skipping fig2.")
    print("\nAll figures written to:", OUT)
    sys.exit(0)

METHOD_PLOT_NAMES = [r'$\nabla_\theta^{{\rm SB}_x} C$',
                     r'$\nabla_\theta^{{\rm SB}_y} C$',
                     r'$\nabla_\theta C$']
METHOD_COLORS = ['darkorange', 'steelblue', 'forestgreen']   # SB_x, SB_y, exact
g2_idx = [gi for gi, g in enumerate(g_values) if g > 0]
nR2 = len(g2_idx)

fig, axes = plt.subplots(nR2, nK, figsize=(5.6*nK, 3.4*nR2), sharex=True)
LW2 = 3.0
# exact (mi=2) on top and dashed, so it sits above the two SB approximations while
# the dashes still let them show through where the curves coincide.
def _style(mi):
    return dict(color=METHOD_COLORS[mi], lw=LW2,
                ls=('--' if mi == 2 else '-'), zorder=3+mi)
for r, gi in enumerate(g2_idx):
    g = g_values[gi]
    for ki in range(nK):
        ax = axes[r, ki]
        for mi in (0, 1, 2):        # SB_x, SB_y, exact (exact drawn last → on top)
            ax.plot(epochs, np.nanmean(cmp_loss[mi,ki,gi], axis=0), **_style(mi))
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(epochs[0], epochs[-1]); ax.grid(alpha=0.25)
        if r == 0: ax.set_title(kerr_names[ki], fontsize=25)
        if r == nR2-1: ax.set_xlabel("Epoch")
        if ki == 0: ax.set_ylabel(fr"$g/\bar\kappa={g/kbar:.2f}$" + "\nMSE")
        if r == 0 and ki == nK-1:
            hs = [Line2D([0],[0], **_style(mi)) for mi in (0, 1, 2)]
            ax.legend(hs, METHOD_PLOT_NAMES, loc='upper right', fontsize=16)
fig.tight_layout()
_save(fig, "training_v4_fig2_method")

print("\nAll figures written to:", OUT)
