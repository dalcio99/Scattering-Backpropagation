"""
run_sweep_dual.py
Quasi-symmetry and gradient-approximation quality vs g and N.
Dual x-axis: primary = g|a_in|²/κ² (swept), secondary = g|ā|²/κ (a posteriori).

6 conditions: 3 nonlinearities * 2 J-coupling variances
  Nonlinearities : self-Kerr, nearest-neighbour cross-Kerr, all-to-all cross-Kerr
  J variance     : Case 1 (σ²=1),  Case 2 (σ²=1/N)

compute_angles returns 9 values:
  cos_Sx, angle_Sx  — Frobenius cosine/angle between S and sigma_x S^T sigma_x
  cos_Sz, angle_Sz  — same for sigma_z
  cos_Gx, angle_Gx  — cosine/angle between true_grad and sb_grad (U=sigma_x)
  cos_Gz, angle_Gz  — cosine/angle between true_grad and sb_grad (U=sigma_z)
  n_ss              — mean photon number per cavity at steady state

Run with:  conda run -n jax-env python run_sweep_dual.py
Outputs land in the same directory as this script.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.optimize import fsolve
import jax
import jax.numpy as jnp
from jax import jacfwd
import warnings
warnings.filterwarnings('ignore')

jax.config.update('jax_enable_x64', True)
print('JAX devices:', jax.devices())

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# parameters 
# ---------------------------------------------------------------------------
N_values  = np.array([5, 10, 15, 20, 25, 30, 35, 40])

g_low    = np.linspace(0.0, 1.0, 11)          # 0.0, 0.1, ..., 1.0
g_high   = np.linspace(1.5, 8.0, 14)          # 1.5, 2.0, ..., 8.0
g_values = np.concatenate([g_low, g_high])    # 25 points total

num_iters = 30
kappa_val = 1.0

def output_idx(N):
    return N // 2

print(f'N values  : {N_values}')
print(f'g values  : {np.round(g_values, 2)}')
print(f'MC iters  : {num_iters}')

# ---------------------------------------------------------------------------
# G matrices 
# ---------------------------------------------------------------------------
def build_G_self(N):
    return np.eye(N)

def build_G_cross_nn(N):
    G = np.zeros((N, N))
    for i in range(N):
        G[i, (i - 1) % N] = 1
        G[i, (i + 1) % N] = 1
    return G

def build_G_alltoall(N):
    return np.ones((N, N))

# ---------------------------------------------------------------------------
# J sampling
# ---------------------------------------------------------------------------
def sample_J(N, case):
    sigma = 1.0 if case == 1 else 1.0 / np.sqrt(N)
    R = sigma * np.random.randn(N, N)
    U = np.triu(R)
    return U + U.T - np.diag(np.diag(U))

# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def frobenius_cosine(A, B):
    ip = np.real(np.trace(A.T @ B))
    nA = np.linalg.norm(A, 'fro')
    nB = np.linalg.norm(B, 'fro')
    if nA < 1e-14 or nB < 1e-14:
        return np.nan
    return float(np.clip(ip / (nA * nB), -1.0, 1.0))

def cosine_similarity(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-14 or nv < 1e-14:
        return np.nan
    return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))

def cos_to_deg(c):
    if np.isnan(c):
        return np.nan
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def compute_angles(N, g, G, J_case):
    """
    Returns (cos_Sx, angle_Sx, cos_Sz, angle_Sz, cos_Gx, angle_Gx, cos_Gz, angle_Gz, n_ss).
    n_ss = mean photon number per cavity at steady state (g * n_ss / kappa = g_eff).
    Returns (nan,)*9 on solver failure.
    """
    kappa = np.ones(N) * kappa_val
    J = sample_J(N, J_case)
    OUTPUT_IDX = output_idx(N)

    theta = np.random.uniform(0, 2*np.pi, size=(N,))
    x_in = np.cos(theta) * np.sqrt(2)
    p_in = np.sin(theta) * np.sqrt(2)

    I_N   = np.eye(N);   O_N = np.zeros((N, N))
    sig_x = np.block([[O_N, I_N], [I_N,  O_N]])
    sig_z = np.block([[I_N, O_N], [O_N, -I_N]])

    def F(y):
        x, p = y[:N], y[N:]
        r  = G @ (x**2 + p**2)
        Fx = -0.5*kappa*x + J@p + 0.5*g*p*r - np.sqrt(kappa)*x_in
        Fp = -0.5*kappa*p - J@x - 0.5*g*x*r - np.sqrt(kappa)*p_in
        return np.concatenate([Fx, Fp])

    starts = [
        np.zeros(2*N),
        0.1 * np.random.randn(2*N),
        0.5 * np.random.randn(2*N),
        np.concatenate([x_in, p_in]) * 0.5,
    ]
    best_res, best_sol = np.inf, None
    for y0 in starts:
        sol, info, ier, _ = fsolve(F, y0, full_output=True)
        res = np.max(np.abs(info['fvec']))
        if ier == 1 and res < best_res:
            best_res, best_sol = res, sol
    if best_res > 1e-8:
        return (np.nan,) * 9

    sol    = best_sol
    x_ss   = sol[:N]
    p_ss   = sol[N:]
    n_ss   = float(np.mean((x_ss**2 + p_ss**2) / 2))   # mean photons per cavity

    def F_jax(y, J_):
        x, p = y[:N], y[N:]
        r  = G @ (x**2 + p**2)
        Fx = -0.5*kappa*x + J_ @ p + 0.5*g*p*r - jnp.sqrt(kappa)*x_in
        Fp = -0.5*kappa*p - J_ @ x - 0.5*g*x*r - jnp.sqrt(kappa)*p_in
        return jnp.concatenate([Fx, Fp])

    try:
        Jac     = np.array(jacfwd(F_jax, argnums=0)(jnp.array(sol), jnp.array(J)))
        inv_Jac = np.linalg.inv(Jac)
    except Exception:
        return (np.nan,) * 9

    k_sqrt     = np.diag(np.tile(np.sqrt(kappa), 2))
    k_sqrt_inv = np.diag(np.tile(1.0 / np.sqrt(kappa), 2))
    S          = np.eye(2*N) + k_sqrt @ inv_Jac @ k_sqrt

    def _s_cosine(U):
        S_approx = U @ S.T @ np.linalg.inv(U)
        return frobenius_cosine(S, S_approx)

    cos_Sx = _s_cosine(sig_x);  angle_Sx = cos_to_deg(cos_Sx)
    cos_Sz = _s_cosine(sig_z);  angle_Sz = cos_to_deg(cos_Sz)

    if abs(x_ss[OUTPUT_IDX]) < 1e-10:
        return cos_Sx, angle_Sx, cos_Sz, angle_Sz, np.nan, np.nan, np.nan, np.nan, n_ss

    x_out    = x_in + np.sqrt(kappa) * x_ss  # sqrt(kappa) per input-output relation (all published runs used kappa = 1)
    dCdx_out = np.zeros(2*N)
    dCdx_out[OUTPUT_IDX] = 2.0 * x_out[OUTPUT_IDX]

    dFdJ = jax.jacrev(F_jax, argnums=1)(jnp.array(sol), jnp.array(J))

    true_grad = -dFdJ.T @ inv_Jac.T @ k_sqrt @ dCdx_out

    def _sb_grad(U_):
        delta_in  = np.linalg.inv(U_) @ dCdx_out  # error signal beta*U^-1 grad C (beta absorbed; published runs used kappa = 1)
        delta_out = S @ delta_in
        return -dFdJ.T @ k_sqrt_inv @ U_ @ (delta_out - delta_in)

    sb_x = _sb_grad(sig_x)
    sb_z = _sb_grad(sig_z)

    cos_Gx = cosine_similarity(true_grad.ravel(), sb_x.ravel())
    cos_Gz = cosine_similarity(true_grad.ravel(), sb_z.ravel())
    angle_Gx = cos_to_deg(cos_Gx)
    angle_Gz = cos_to_deg(cos_Gz)

    return cos_Sx, angle_Sx, cos_Sz, angle_Sz, cos_Gx, angle_Gx, cos_Gz, angle_Gz, n_ss

# ---------------------------------------------------------------------------
# Different cases
# ---------------------------------------------------------------------------
conditions = [
    ("Self-Kerr",               r"$\sigma^2=1$",   build_G_self,     1),
    ("Self-Kerr",               r"$\sigma^2=1/N$", build_G_self,     2),
    ("Cross-Kerr (n.n.)",       r"$\sigma^2=1$",   build_G_cross_nn, 1),
    ("Cross-Kerr (n.n.)",       r"$\sigma^2=1/N$", build_G_cross_nn, 2),
    ("Cross-Kerr (all-to-all)", r"$\sigma^2=1$",   build_G_alltoall, 1),
    ("Cross-Kerr (all-to-all)", r"$\sigma^2=1/N$", build_G_alltoall, 2),
]

nC = len(conditions)
nN = len(N_values)
ng = len(g_values)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
shape = (nC, nN, ng)

mean_cosSx = np.full(shape, np.nan);  std_cosSx = np.full(shape, np.nan)
mean_angSx = np.full(shape, np.nan);  std_angSx = np.full(shape, np.nan)
mean_cosSz = np.full(shape, np.nan);  std_cosSz = np.full(shape, np.nan)
mean_angSz = np.full(shape, np.nan);  std_angSz = np.full(shape, np.nan)
mean_cosGx = np.full(shape, np.nan);  std_cosGx = np.full(shape, np.nan)
mean_angGx = np.full(shape, np.nan);  std_angGx = np.full(shape, np.nan)
mean_cosGz = np.full(shape, np.nan);  std_cosGz = np.full(shape, np.nan)
mean_angGz = np.full(shape, np.nan);  std_angGz = np.full(shape, np.nan)
succ_frac  = np.zeros(shape)

raw_cosSx = np.full((nC, nN, ng, num_iters), np.nan)
raw_cosSz = np.full((nC, nN, ng, num_iters), np.nan)
raw_cosGx = np.full((nC, nN, ng, num_iters), np.nan)
raw_cosGz = np.full((nC, nN, ng, num_iters), np.nan)

# n_ss per sample → g_eff = g * n_ss / kappa a posteriori
raw_nss  = np.full((nC, nN, ng, num_iters), np.nan)

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
for ci, (lk, lj, G_fn, jcase) in enumerate(conditions):
    print(f"\n{'='*60}")
    print(f"Condition {ci+1}/{nC} :  {lk},  {lj}")
    print(f"{'='*60}")
    for ni, N in enumerate(N_values):
        G = G_fn(N)
        for gi, g in enumerate(g_values):
            np.random.seed(1000*ci + 100*ni + gi)
            cSx_l, aSx_l = [], []
            cSz_l, aSz_l = [], []
            cGx_l, aGx_l = [], []
            cGz_l, aGz_l = [], []

            for it in range(num_iters):
                res = compute_angles(N, g, G, jcase)
                cSx, aSx, cSz, aSz, cGx, aGx, cGz, aGz, nss = res
                raw_cosSx[ci, ni, gi, it] = cSx
                raw_cosSz[ci, ni, gi, it] = cSz
                raw_cosGx[ci, ni, gi, it] = cGx
                raw_cosGz[ci, ni, gi, it] = cGz
                raw_nss[ci, ni, gi, it]   = nss
                if np.isfinite(cSx): cSx_l.append(cSx); aSx_l.append(aSx)
                if np.isfinite(cSz): cSz_l.append(cSz); aSz_l.append(aSz)
                if np.isfinite(cGx): cGx_l.append(cGx); aGx_l.append(aGx)
                if np.isfinite(cGz): cGz_l.append(cGz); aGz_l.append(aGz)

            succ_frac[ci, ni, gi] = len(cSx_l) / num_iters
            if len(cSx_l) >= 3:
                mean_cosSx[ci,ni,gi] = np.mean(cSx_l); std_cosSx[ci,ni,gi] = np.std(cSx_l)
                mean_angSx[ci,ni,gi] = np.mean(aSx_l); std_angSx[ci,ni,gi] = np.std(aSx_l)
            if len(cSz_l) >= 3:
                mean_cosSz[ci,ni,gi] = np.mean(cSz_l); std_cosSz[ci,ni,gi] = np.std(cSz_l)
                mean_angSz[ci,ni,gi] = np.mean(aSz_l); std_angSz[ci,ni,gi] = np.std(aSz_l)
            if len(cGx_l) >= 3:
                mean_cosGx[ci,ni,gi] = np.mean(cGx_l); std_cosGx[ci,ni,gi] = np.std(cGx_l)
                mean_angGx[ci,ni,gi] = np.mean(aGx_l); std_angGx[ci,ni,gi] = np.std(aGx_l)
            if len(cGz_l) >= 3:
                mean_cosGz[ci,ni,gi] = np.mean(cGz_l); std_cosGz[ci,ni,gi] = np.std(cGz_l)
                mean_angGz[ci,ni,gi] = np.mean(aGz_l); std_angGz[ci,ni,gi] = np.std(aGz_l)

        n_valid = np.sum(np.isfinite(mean_cosSx[ci, ni, :]))
        avg_sr  = succ_frac[ci, ni, :].mean()
        print(f"  N={N:2d}  |  Sx-cos valid: {n_valid:2d}/{ng}  "
              f"|  avg convergence: {avg_sr:.2f}")

print("\n✓  Sweep complete.")

# derive raw angle arrays from raw cosines
raw_angSx = np.degrees(np.arccos(np.clip(raw_cosSx, -1.0, 1.0)))
raw_angSz = np.degrees(np.arccos(np.clip(raw_cosSz, -1.0, 1.0)))
raw_angGx = np.degrees(np.arccos(np.clip(raw_cosGx, -1.0, 1.0)))
raw_angGz = np.degrees(np.arccos(np.clip(raw_cosGz, -1.0, 1.0)))

# mean g_eff = g * n_ss / kappa, averaged over MC realisations per (ci, ni, gi)
# shape: (nC, nN, ng)
mean_geff = np.full(shape, np.nan)
for ci in range(nC):
    for ni in range(nN):
        for gi, g in enumerate(g_values):
            nss_vals = raw_nss[ci, ni, gi, :]
            finite_nss = nss_vals[np.isfinite(nss_vals)]
            if len(finite_nss) > 0:
                mean_geff[ci, ni, gi] = g * np.mean(finite_nss) / kappa_val

np.savez(os.path.join(OUT, "angle_stats_dual.npz"),
         mean_cosSx=mean_cosSx, std_cosSx=std_cosSx,
         mean_angSx=mean_angSx, std_angSx=std_angSx,
         mean_cosSz=mean_cosSz, std_cosSz=std_cosSz,
         mean_angSz=mean_angSz, std_angSz=std_angSz,
         mean_cosGx=mean_cosGx, std_cosGx=std_cosGx,
         mean_angGx=mean_angGx, std_angGx=std_angGx,
         mean_cosGz=mean_cosGz, std_cosGz=std_cosGz,
         mean_angGz=mean_angGz, std_angGz=std_angGz,
         raw_cosSx=raw_cosSx, raw_cosSz=raw_cosSz,
         raw_cosGx=raw_cosGx, raw_cosGz=raw_cosGz,
         raw_nss=raw_nss, mean_geff=mean_geff,
         succ_frac=succ_frac,
         N_values=N_values, g_values=g_values)
print("Saved  angle_stats_dual.npz")

# ---------------------------------------------------------------------------
#  PLOTTING
# ---------------------------------------------------------------------------

cmap_N = plt.cm.plasma_r
norm_N = mcolors.Normalize(vmin=N_values.min(), vmax=N_values.max())

g_ticks    = np.array([0, 0.5, 1.0, 2, 4, 6, 8])
g_tick_pos = [np.argmin(np.abs(g_values - v)) for v in g_ticks]

def _apply_gaxis(ax):
    ax.set_xticks(g_tick_pos)
    ax.set_xticklabels([str(v) for v in g_ticks], fontsize=8)
    ax.set_xlabel(r"$g|\bar{a}_{\rm in}|^2/\kappa^2$", fontsize=10)

# ---------------------------------------------------------------------------
# Heatmap Helper
# ---------------------------------------------------------------------------
def heatmap_figure(data, suptitle, cmap, vmin, vmax):
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    fig.suptitle(suptitle, fontsize=13, y=1.01)
    for ci, (lk, lj, _, _) in enumerate(conditions):
        row, col = ci // 2, ci % 2
        ax = axes[row, col]
        im = ax.imshow(
            data[ci], cmap=cmap, vmin=vmin, vmax=vmax,
            aspect='auto', origin='lower',
            extent=[-0.5, ng - 0.5, -0.5, nN - 0.5]
        )
        _apply_gaxis(ax)
        ax.set_yticks(np.arange(nN))
        ax.set_yticklabels(N_values, fontsize=8)
        ax.set_ylabel(r"$N$", fontsize=10)
        ax.set_title(f"{lk}   {lj}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

def _save_heatmap(data, fname, title, cmap, vmin, vmax):
    fig = heatmap_figure(data, title, cmap, vmin, vmax)
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved  {fname}")

# ---------------------------------------------------------------------------
# Mean cosine-similarity heatmap
# ---------------------------------------------------------------------------
_save_heatmap(mean_cosSx, "heatmap_Sx_cosine.png",
    r"$\cos(S,\; \sigma_x S^T \sigma_x)$  [Frobenius cosine similarity]",
    'viridis', -1, 1)
_save_heatmap(mean_cosSz, "heatmap_Sz_cosine.png",
    r"$\cos(S,\; \sigma_z S^T \sigma_z)$  [Frobenius cosine similarity]",
    'viridis', -1, 1)
_save_heatmap(mean_cosGx, "heatmap_Gx_cosine.png",
    r"$\cos(\nabla_{\rm true},\; \nabla_{\rm SB}^{\sigma_x})$  [$\beta\to0$]",
    'viridis', -1, 1)
_save_heatmap(mean_cosGz, "heatmap_Gz_cosine.png",
    r"$\cos(\nabla_{\rm true},\; \nabla_{\rm SB}^{\sigma_z})$  [$\beta\to0$]",
    'viridis', -1, 1)

# ---------------------------------------------------------------------------
# Mean angle heatmaps
# ---------------------------------------------------------------------------
_save_heatmap(mean_angSx, "heatmap_Sx_angle.png",
    r"$\angle(S,\; \sigma_x S^T \sigma_x)$  [Frobenius, degrees]",
    'viridis_r', 0, 90)
_save_heatmap(mean_angSz, "heatmap_Sz_angle.png",
    r"$\angle(S,\; \sigma_z S^T \sigma_z)$  [Frobenius, degrees]",
    'viridis_r', 0, 90)
_save_heatmap(mean_angGx, "heatmap_Gx_angle.png",
    r"$\angle(\nabla_{\rm true},\; \nabla_{\rm SB}^{\sigma_x})$  [degrees, $\beta\to0$]",
    'inferno_r', 0, 90)
_save_heatmap(mean_angGz, "heatmap_Gz_angle.png",
    r"$\angle(\nabla_{\rm true},\; \nabla_{\rm SB}^{\sigma_z})$  [degrees, $\beta\to0$]",
    'inferno_r', 0, 90)

# ---------------------------------------------------------------------------
# std cosine-similarity heatmaps
# ---------------------------------------------------------------------------
_save_heatmap(std_cosSx, "heatmap_std_Sx_cosine.png",
    r"std$[\cos(S,\; \sigma_x S^T \sigma_x)]$", 'magma', 0, 0.3)
_save_heatmap(std_cosSz, "heatmap_std_Sz_cosine.png",
    r"std$[\cos(S,\; \sigma_z S^T \sigma_z)]$", 'magma', 0, 0.3)
_save_heatmap(std_cosGx, "heatmap_std_Gx_cosine.png",
    r"std$[\cos(\nabla_{\rm true},\; \nabla_{\rm SB}^{\sigma_x})]$", 'magma', 0, 0.5)
_save_heatmap(std_cosGz, "heatmap_std_Gz_cosine.png",
    r"std$[\cos(\nabla_{\rm true},\; \nabla_{\rm SB}^{\sigma_z})]$", 'magma', 0, 0.5)

print("Saved  12 heatmaps")

# ---------------------------------------------------------------------------
# Scatter helpers
# ---------------------------------------------------------------------------
def _draw_faded(ax, gs, vals, stds, conv, color, std_scale, lw=1.4):
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
        sz = 8 + 80 * float(min(max(s, 0), std_scale) / std_scale) if np.isfinite(s) else 8
        ax.scatter(gs[gi], vals[gi], color=color, s=sz,
                   alpha=float(alpha_pts[gi]), zorder=3, linewidths=0)

def _draw_errband(ax, gs, vals, stds, color, lw=1.4, ms=3.5):
    fin = np.isfinite(vals) & np.isfinite(stds)
    if fin.sum() == 0:
        return
    ax.plot(gs[fin], vals[fin], '-o', color=color, lw=lw, ms=ms, alpha=0.85)
    ax.fill_between(gs[fin], vals[fin] - stds[fin], vals[fin] + stds[fin],
                    color=color, alpha=0.15)

def _add_geff_axis(ax, ci):
    """Add secondary top x-axis showing g_eff = g * <n_ss> / kappa."""
    # mean over N axis to get a representative geff curve vs gi
    geff_vs_gi = np.nanmean(mean_geff[ci], axis=0)  # shape (ng,)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    tick_g_vals = [0, 0.5, 1.0, 2, 4, 6, 8]
    tick_pos, tick_labels = [], []
    for tg in tick_g_vals:
        idx = np.argmin(np.abs(g_values - tg))
        geff_val = geff_vs_gi[idx]
        tick_pos.append(tg)
        tick_labels.append(f"{geff_val:.2g}" if np.isfinite(geff_val) else "")
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels, fontsize=6)
    ax2.set_xlabel(r"$g|\bar{a}|^2/\kappa$", fontsize=7, labelpad=2)
    return ax2

def _scatter_main(metrics, suptitle, fname, ylim, yticks, mode='faded'):
    nM = len(metrics)
    fig, axes = plt.subplots(nM, nC, figsize=(4*nC, 3.5*nM), sharey='row')
    if nM == 1:
        axes = axes[np.newaxis, :]

    for mi, (mdata, sdata, ylabel, std_scale) in enumerate(metrics):
        for ci, (lk, lj, _, _) in enumerate(conditions):
            ax = axes[mi, ci]
            for ni, N in enumerate(N_values):
                color = cmap_N(norm_N(N))
                vals  = mdata[ci, ni, :]
                stds  = sdata[ci, ni, :]
                conv  = succ_frac[ci, ni, :]
                if not np.any(np.isfinite(vals)):
                    continue
                if mode == 'faded':
                    _draw_faded(ax, g_values, vals, stds, conv, color, std_scale)
                else:
                    _draw_errband(ax, g_values, vals, stds, color)
            ax.axvline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
            ax.set_xlabel(r"$g|\bar{a}_{\rm in}|^2/\kappa^2$", fontsize=9)
            ax.set_ylim(*ylim); ax.set_yticks(yticks)
            ax.set_xticks([0, 2, 4, 6, 8]); ax.grid(alpha=0.25)
            if ci == 0: ax.set_ylabel(ylabel, fontsize=9)
            if mi == 0:
                ax.set_title(f"{lk}\n{lj}", fontsize=8)
                _add_geff_axis(ax, ci)

    sm = plt.cm.ScalarMappable(cmap=cmap_N, norm=norm_N)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label=r"$N$", fraction=0.012, pad=0.02)
    cbar.set_ticks(N_values)
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved  {fname}")

# ---------------------------------------------------------------------------
# cosine-similarity scatter plots
# ---------------------------------------------------------------------------
cos_metrics = [
    (mean_cosSx, std_cosSx, r"$\cos(S,\; \sigma_x S^T \sigma_x)$",                        0.3),
    (mean_cosSz, std_cosSz, r"$\cos(S,\; \sigma_z S^T \sigma_z)$",                        0.3),
    (mean_cosGx, std_cosGx, r"$\cos(\nabla_{\rm true},\;\nabla_{\rm SB}^{\sigma_x})$",    0.4),
    (mean_cosGz, std_cosGz, r"$\cos(\nabla_{\rm true},\;\nabla_{\rm SB}^{\sigma_z})$",    0.4),
]
_scatter_main(cos_metrics,
    r"Cosine similarity vs $g$  [alpha $\propto$ convergence,  dot size $\propto$ std]",
    "scatter_cosine_faded.png", (-1.05, 1.05), [-1, -0.5, 0, 0.5, 1], mode='faded')
_scatter_main(cos_metrics,
    r"Cosine similarity vs $g$  [$\pm$std band]",
    "scatter_cosine_errband.png", (-1.05, 1.05), [-1, -0.5, 0, 0.5, 1], mode='errband')

# ---------------------------------------------------------------------------
# angle scatter plots
# ---------------------------------------------------------------------------
ang_metrics = [
    (mean_angSx, std_angSx, r"$\angle(S,\; \sigma_x S^T \sigma_x)$ [°]",                        30),
    (mean_angSz, std_angSz, r"$\angle(S,\; \sigma_z S^T \sigma_z)$ [°]",                        30),
    (mean_angGx, std_angGx, r"$\angle(\nabla_{\rm true},\;\nabla_{\rm SB}^{\sigma_x})$ [°]",    40),
    (mean_angGz, std_angGz, r"$\angle(\nabla_{\rm true},\;\nabla_{\rm SB}^{\sigma_z})$ [°]",    40),
]
_scatter_main(ang_metrics,
    r"Angles vs $g$  [alpha $\propto$ convergence,  dot size $\propto$ std]",
    "scatter_angles_faded.png", (0, 92), [0, 30, 60, 90], mode='faded')
_scatter_main(ang_metrics,
    r"Angles vs $g$  [$\pm$std band]",
    "scatter_angles_errband.png", (0, 92), [0, 30, 60, 90], mode='errband')

# ---------------------------------------------------------------------------
# Ditribution violins helpers
# ---------------------------------------------------------------------------
G_SUB_IDX  = np.unique(np.round(np.linspace(0, ng - 1, 10)).astype(int))
G_SUB_VALS = g_values[G_SUB_IDX]
N_SEL_IDX  = [0, nN // 2, nN - 1]
N_SEL_VALS = N_values[N_SEL_IDX]
N_SEL_COLS = [cmap_N(norm_N(N)) for N in N_SEL_VALS]

def _add_violins(ax, vdata, vpos, vcols):
    if not vdata:
        return
    try:
        parts = ax.violinplot(vdata, positions=vpos,
                              widths=0.72, showmedians=True, showextrema=False)
        for body, col in zip(parts['bodies'], vcols):
            body.set_facecolor(col); body.set_alpha(0.65); body.set_edgecolor('none')
        parts['cmedians'].set_color('white'); parts['cmedians'].set_linewidth(1.5)
    except Exception:
        for d, p, c in zip(vdata, vpos, vcols):
            ax.scatter([p] * len(d), d, color=c, alpha=0.4, s=8, linewidths=0)

def _violin_vs_g(raw_data, suptitle, fname, ylim, ylabel):
    n_sel   = len(N_SEL_IDX)
    spacing = n_sel + 0.8

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle(suptitle, fontsize=13, y=1.01)

    for ci, (lk, lj, _, _) in enumerate(conditions):
        row, col = ci // 2, ci % 2
        ax = axes[row, col]
        vdata, vpos, vcols    = [], [], []
        group_ticks, g_labels = [], []
        # also build secondary axis labels (geff) for the subsampled g points
        geff_labels = []

        for g_enum, gi in enumerate(G_SUB_IDX):
            base = g_enum * spacing
            group_ticks.append(base + (n_sel - 1) / 2.0)
            g_labels.append(f"{G_SUB_VALS[g_enum]:.1f}")
            geff_mean = np.nanmean(mean_geff[ci, :, gi])
            geff_labels.append(f"{geff_mean:.2g}" if np.isfinite(geff_mean) else "")
            for k, ni in enumerate(N_SEL_IDX):
                s = raw_data[ci, ni, gi, :]
                s = s[np.isfinite(s)]
                if len(s) < 2:
                    continue
                vdata.append(s); vpos.append(base + k); vcols.append(N_SEL_COLS[k])

        _add_violins(ax, vdata, vpos, vcols)
        ax.set_xticks(group_ticks); ax.set_xticklabels(g_labels, fontsize=7)
        ax.set_xlabel(r"$g|\bar{a}_{\rm in}|^2/\kappa^2$", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9); ax.set_ylim(*ylim)
        ax.grid(axis='y', alpha=0.25); ax.set_title(f"{lk}   {lj}", fontsize=10)

        # secondary x-axis showing g_eff
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(group_ticks)
        ax2.set_xticklabels(geff_labels, fontsize=6)
        ax2.set_xlabel(r"$g|\bar{a}|^2/\kappa$", fontsize=7, labelpad=2)

        if ci == 0:
            ax.legend(handles=[Patch(facecolor=N_SEL_COLS[k], alpha=0.7,
                                     label=f"N={N_SEL_VALS[k]}")
                                for k in range(n_sel)],
                      fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"Saved  {fname}")

def _violin_vs_N(raw_data, suptitle, fname, ylim, ylabel):
    g_low_mask  = g_values <= 1.0
    g_high_mask = g_values >  1.0
    regimes = [(g_low_mask,  r"$g \leq 1$", 'steelblue'),
               (g_high_mask, r"$g > 1$",    'tomato')]
    n_reg   = len(regimes)
    spacing = n_reg + 0.8

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle(suptitle, fontsize=13, y=1.01)

    for ci, (lk, lj, _, _) in enumerate(conditions):
        row, col = ci // 2, ci % 2
        ax = axes[row, col]
        vdata, vpos, vcols    = [], [], []
        group_ticks, n_labels = [], []

        for n_enum, ni in enumerate(range(nN)):
            base = n_enum * spacing
            group_ticks.append(base + (n_reg - 1) / 2.0)
            n_labels.append(str(N_values[ni]))
            for r_idx, (gmask, glabel, gcol) in enumerate(regimes):
                s = raw_data[ci, ni, :, :][gmask, :].ravel()
                s = s[np.isfinite(s)]
                if len(s) < 2:
                    continue
                vdata.append(s); vpos.append(base + r_idx); vcols.append(gcol)

        _add_violins(ax, vdata, vpos, vcols)
        ax.set_xticks(group_ticks); ax.set_xticklabels(n_labels, fontsize=8)
        ax.set_xlabel(r"$N$", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9); ax.set_ylim(*ylim)
        ax.grid(axis='y', alpha=0.25); ax.set_title(f"{lk}   {lj}", fontsize=10)
        if ci == 0:
            ax.legend(handles=[Patch(facecolor=gcol, alpha=0.7, label=glabel)
                                for _, glabel, gcol in regimes],
                      fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"Saved  {fname}")

# ---------------------------------------------------------------------------
# Distribution figures
# ---------------------------------------------------------------------------
dist_specs = [
    (raw_cosSx, raw_angSx,
     r"cos$(S,\; \sigma_x S^T \sigma_x)$",
     r"$\angle(S,\; \sigma_x S^T \sigma_x)$", "Sx"),
    (raw_cosSz, raw_angSz,
     r"cos$(S,\; \sigma_z S^T \sigma_z)$",
     r"$\angle(S,\; \sigma_z S^T \sigma_z)$", "Sz"),
    (raw_cosGx, raw_angGx,
     r"cos$(\nabla_{\rm true},\nabla_{\rm SB}^{\sigma_x})$",
     r"$\angle(\nabla_{\rm true},\nabla_{\rm SB}^{\sigma_x})$", "Gx"),
    (raw_cosGz, raw_angGz,
     r"cos$(\nabla_{\rm true},\nabla_{\rm SB}^{\sigma_z})$",
     r"$\angle(\nabla_{\rm true},\nabla_{\rm SB}^{\sigma_z})$", "Gz"),
]

for raw_cos, raw_ang, cos_lbl, ang_lbl, prefix in dist_specs:
    _violin_vs_g(raw_cos,
        f"Distribution of {cos_lbl} vs $g$  "
        f"[N={N_SEL_VALS[0]}, {N_SEL_VALS[1]}, {N_SEL_VALS[2]} side-by-side]",
        f"dist_cos{prefix}_vs_g.png", (-1.05, 1.05), "cosine similarity")
    _violin_vs_N(raw_cos,
        f"Distribution of {cos_lbl} vs $N$  [pooled by $g$-regime]",
        f"dist_cos{prefix}_vs_N.png", (-1.05, 1.05), "cosine similarity")
    _violin_vs_g(raw_ang,
        f"Distribution of {ang_lbl} vs $g$  "
        f"[N={N_SEL_VALS[0]}, {N_SEL_VALS[1]}, {N_SEL_VALS[2]} side-by-side]",
        f"dist_ang{prefix}_vs_g.png", (-2, 92), "angle [°]")
    _violin_vs_N(raw_ang,
        f"Distribution of {ang_lbl} vs $N$  [pooled by $g$-regime]",
        f"dist_ang{prefix}_vs_N.png", (-2, 92), "angle [°]")

# ---------------------------------------------------------------------------
# Convergence heatmap
# ---------------------------------------------------------------------------
_save_heatmap(succ_frac, "heatmap_convergence.png",
    "fsolve convergence rate (fraction of 30 runs)", 'RdYlGn', 0, 1)

print("\nAll outputs written to:", OUT)
