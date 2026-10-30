"""
plot_quasi_symmetry_paper.py

Single-panel figure of the scattering-matrix quasi-symmetry cosine
for the MNIST Kerr network, built from the existing sweep data
(quasi_symmetry_results.npz, g in {0, 0.1, 0.2, 0.3, 0.4}, 50 images).

What is plotted (cosine only, no angle):
    cos(S^dagger, sigma_x S sigma_x)   <- array cos_z_all  (code sig_z = a-basis sigma_x)
    cos(S^dagger, sigma_y S sigma_y)   <- array cos_x_all  (code sig_x = a-basis sigma_y)

Dual x-axis (same convention as Angle_stats/paper_figs_dual/plot_paper.py):
    primary (bottom): g <|a_in|^2>_avg / kappabar^2   (deterministic from images)
    secondary (top) : g <|abar|^2>_avg / kappabar      (= chi_ss, from the sweep)

Convention: a = (x + i p)/sqrt(2)  ->  |a|^2 = (x^2 + p^2)/2.

Run:  /opt/miniconda3/envs/jax-env/bin/python plot_quasi_symmetry_paper.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# paper style (large fonts / markers)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size'       : 22,
    'axes.titlesize'  : 22,
    'axes.labelsize'  : 26,
    'xtick.labelsize' : 20,
    'ytick.labelsize' : 20,
    'legend.fontsize' : 22,
    'axes.linewidth'  : 1.5,
    'xtick.major.size': 7,
    'ytick.major.size': 7,
    'pdf.fonttype'    : 42,   # editable text in the PDF
    'ps.fonttype'     : 42,
})

# ---------------------------------------------------------------------------
# load sweep results
# ---------------------------------------------------------------------------
d = np.load(os.path.join(SCRIPT_DIR, 'Data', 'quasi_symmetry_results.npz'))
G_VALUES = d['G_VALUES']
chi_ss_all = d['chi_ss_all']     # (nG, nImg)  = g<|abar|^2>/kappa_eff  (secondary axis)
cos_x_all = d['cos_x_all']      # code sig_x = a-basis sigma_y
cos_z_all = d['cos_z_all']      # code sig_z = a-basis sigma_x
nG = len(G_VALUES)

# ---------------------------------------------------------------------------
# input photon number for the primary axis (deterministic from images)
# ---------------------------------------------------------------------------
# g <|a_in|^2>_avg / kappabar^2 ; kappa = kappa1 = 1 -> kappabar = 1.
KAPPA_BAR    = 1.0
INPUT_PIXELS = 784
imgs = np.load(os.path.join(SCRIPT_DIR, 'Data', 'mnist_test_imgs.npy')).astype(np.float64)
J = np.load(os.path.join(SCRIPT_DIR, 'J_mnist_seed41.npy'))
N = J.shape[0]
N_IMAGES = cos_x_all.shape[1]

# per-image mean input photon number over N modes: <|a_in|^2>_avg, p_in = 0
a_in_per_img = np.empty(N_IMAGES)
for si in range(N_IMAGES):
    x_in = np.zeros(N); x_in[:INPUT_PIXELS] = imgs[si].flatten() / 100.0
    a_in_per_img[si] = np.mean((x_in ** 2) / 2.0)
a_in_avg = float(np.mean(a_in_per_img))          # image-averaged <|a_in|^2>_avg

# ---------------------------------------------------------------------------
# x coordinates (primary axis) and secondary-axis (chi_ss) labels 
# ---------------------------------------------------------------------------
x_prim   = G_VALUES * a_in_avg / (KAPPA_BAR ** 2)          # g<|a_in|^2>_avg/kbar^2
chi_mean = np.array([np.nanmean(chi_ss_all[gi]) for gi in range(nG)])  # g<|abar|^2>/kbar

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 7.5), constrained_layout=True)

series = [
    (cos_z_all, r"$\cos(S^\dagger,\;\sigma_x S\,\sigma_x)$", '#1f5fa8', 'o'),  # a-basis sigma_x
    (cos_x_all, r"$\cos(S^\dagger,\;\sigma_y S\,\sigma_y)$", '#c0392b', 's'),  # a-basis sigma_y
]

for cos_all, label, color, marker in series:
    cm = np.array([np.nanmean(cos_all[gi]) for gi in range(nG)])
    cs = np.array([np.nanstd(cos_all[gi])  for gi in range(nG)])
    # faint per-image scatter
    for gi in range(nG):
        vals = cos_all[gi][np.isfinite(cos_all[gi])]
        ax.scatter(np.full(len(vals), x_prim[gi]), vals,
                   s=26, color=color, alpha=0.18, zorder=2, linewidths=0)
    # mean +/- std
    ax.errorbar(x_prim, cm, yerr=cs, fmt=f'{marker}-', color=color,
                lw=2.6, ms=11, capsize=6, capthick=2.0, zorder=5, label=label)

ax.set_xlabel(r"$g\,\langle |a_{\mathrm{in}}|^2 \rangle_{\mathrm{avg}}/\bar{\kappa}^2$")
ax.set_ylabel(r"cosine similarity")
ax.grid(alpha=0.25)
ax.legend(loc='lower left', frameon=True)

# y-range: zoom on where the data lives (cos in ~[0.95, 1])
lo = np.nanmin([np.nanmean(c[gi]) - np.nanstd(c[gi])
                for c, _, _, _ in series for gi in range(nG)])
ax.set_ylim(min(0.95, lo - 0.01), 1.004)

# secondary top axis: g <|abar|^2>_avg / kappabar  (= chi_ss)
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(x_prim)
ax2.set_xticklabels([f"{v:.3f}" for v in chi_mean])
ax2.tick_params(axis='x', pad=6)
ax2.set_xlabel(r"$g\,\langle |\bar{a}|^2 \rangle_{\mathrm{avg}}/\bar{\kappa}$",
               labelpad=12)

out_pdf = os.path.join(SCRIPT_DIR, 'quasi_symmetry_comparison_paper.pdf')
out_png = os.path.join(SCRIPT_DIR, 'quasi_symmetry_comparison_paper.png')
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"N = {N}, images = {N_IMAGES}")
print(f"<|a_in|^2>_avg (image-averaged) = {a_in_avg:.4f}")
print(f"g values            : {np.round(G_VALUES, 3)}")
print(f"primary axis x       : {np.round(x_prim, 4)}")
print(f"secondary axis (chi) : {np.round(chi_mean, 4)}")
print(f"cos(sigma_x) mean    : {np.round([np.nanmean(cos_z_all[gi]) for gi in range(nG)], 4)}")
print(f"cos(sigma_y) mean    : {np.round([np.nanmean(cos_x_all[gi]) for gi in range(nG)], 4)}")
print(f"\nSaved  {out_pdf}")
print(f"Saved  {out_png}")
