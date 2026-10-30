"""
quasi_symmetry_angle.py

Computes the quasi-symmetry angle of the linearized scattering matrix S for the MNIST
Kerr network, for the two SB perturbation types.

U operators (2N x 2N, quadrature x⊕p basis)
--------------------------------------------
    sigma_y (complex) = sigma_x (quadrature):  U_x = [[0, I_N], [I_N, 0]]
        → imaginary input perturbation (δa_in = -iβ err) = p-quadrature perturbation
        → used in noise_rk4.py mode='sigma_y'
        → U_x⁻¹ = U_x  (since U_x² = I)

    sigma_x (complex) = sigma_z (quadrature):  U_z = [[I_N, 0], [0, -I_N]]
        → real input perturbation (δa_in = β err) = x-quadrature perturbation
        → used in noise_rk4.py mode='sigma_x'
        → U_z⁻¹ = U_z  (since U_z² = I)

Both are unitary (U† = U⁻¹), so ‖USU⁻¹‖_F = ‖S‖_F for both.

Computation
-----------
For N=963, 2N=1926: S is a 1926x1926 dense complex matrix (~60 MB).
Computed via  scipy.sparse.linalg.factorized(Jac) + 1926 back-substitutions.
Each image takes ~5-15 s depending on hardware.

Run from the MNIST folder:  python quasi_symmetry_angle.py
"""

import os, sys
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# MNIST images
# ---------------------------------------------------------------------------
_imgs_path = os.path.join(SCRIPT_DIR, 'Data', 'mnist_test_imgs.npy')
_lbls_path = os.path.join(SCRIPT_DIR, 'Data', 'mnist_test_lbls.npy')
if os.path.exists(_imgs_path) and os.path.exists(_lbls_path):
    test_imgs   = np.load(_imgs_path).astype(np.float64)
    test_labels = np.load(_lbls_path).astype(np.int32)
    print(f"MNIST loaded: {test_imgs.shape[0]} test images.")
else:
    print("MNIST .npy not found — using synthetic inputs.")
    rng = np.random.default_rng(42)
    test_imgs   = rng.exponential(33.0, (200, 28, 28)).clip(0, 255)
    test_labels = rng.integers(0, 10, 200)

# ---------------------------------------------------------------------------
# System parameters (matching notebook / regime_diagnostic.py) 
# ---------------------------------------------------------------------------
G_VALUES     = [0.0, 0.1, 0.2, 0.3, 0.4]
KAPPA        = 1.0
KAPPA1       = 1.0
KAPPA_EFF    = (KAPPA + KAPPA1) / 2.0
N_LOGITS     = 10
INPUT_PIXELS = 784
N_IMAGES     = 50

# ---------------------------------------------------------------------------
# Load J (pre-saved from seed 41; generate with params_init.py if missing) 
# ---------------------------------------------------------------------------
J_path = os.path.join(SCRIPT_DIR, 'J_mnist_seed41.npy')
if not os.path.exists(J_path):
    raise FileNotFoundError(
        f"{J_path} not found. Run once with JAX:\n"
        "  python3 -c \"import jax; jax.config.update('jax_enable_x64',True); "
        "import jax.numpy as jnp, numpy as np, sys; sys.path.insert(0,'.'); "
        "from params_init import create_conn_matrix; "
        "key=jax.random.key(41); key,sub,_=jax.random.split(key,3); "
        "J=create_conn_matrix(sub,kernel_sizes=[6,4],strides=[2,2],input_pixels=28,num_logits=10); "
        "np.save('J_mnist_seed41.npy', np.array(J))\"")
J_np = np.load(J_path)
N    = J_np.shape[0]
print(f"N = {N}  (784+144+25+10)  →  2N = {2*N}")

kappa_v  = KAPPA  * np.ones(N)
kappa1_v = KAPPA1 * np.ones(N)
sqrt_k   = np.sqrt(kappa_v)          # = 1 everywhere

J_sp = scipy.sparse.csr_matrix(J_np)
I_2N = scipy.sparse.eye(2*N, format='csc')

# ------------------------------------------------------
#  ODE Jacobian at steady state (SS)
# ------------------------------------------------------
def jac_F_sparse(x_ss, p_ss, g):
    keff = 0.5*(kappa_v + kappa1_v)
    d_xx =  -keff + g*x_ss*p_ss
    d_xp =   0.5*g*(x_ss**2 + 3*p_ss**2)
    d_px =  -0.5*g*(3*x_ss**2 + p_ss**2)
    d_pp =  -keff - g*x_ss*p_ss
    return scipy.sparse.bmat([
        [scipy.sparse.diags(d_xx),          J_sp + scipy.sparse.diags(d_xp)],
        [-J_sp + scipy.sparse.diags(d_px),  scipy.sparse.diags(d_pp)       ]
    ], format='csc')

# ----------------------------------------------------------------
# Sparse Newton solver for SS (same as regime_diagnostic.py) 
# -----------------------------------------------------------------
A_g0 = np.eye(N) + J_np @ J_np
L_g0 = scipy.linalg.lu_factor(A_g0)

def analytic_ss_g0(x_in):
    x = -scipy.linalg.lu_solve(L_g0, x_in)
    p = -J_np @ x
    return np.concatenate([x, p])

def F_ode(y, g, x_in, p_in=None):
    x, p = y[:N], y[N:]
    if p_in is None:
        p_in = np.zeros(N)
    r  = x**2 + p**2
    Fx = (-0.5*(kappa_v+kappa1_v)*x + 0.5*g*r*p + J_np@p - sqrt_k*x_in)
    Fp = (-0.5*(kappa_v+kappa1_v)*p - 0.5*g*r*x - J_np@x - sqrt_k*p_in)
    return np.concatenate([Fx, Fp])

def newton_ss(y0, g, x_in, tol=1e-9, maxiter=30):
    y = y0.copy()
    p_in = np.zeros(N)
    for _ in range(maxiter):
        F   = F_ode(y, g, x_in, p_in)
        res = np.max(np.abs(F))
        if res < tol:
            break
        Jac  = jac_F_sparse(y[:N], y[N:], g)
        dy   = scipy.sparse.linalg.spsolve(Jac, -F)
        for alpha in [1.0, 0.5, 0.25, 0.125]:
            y_new = y + alpha * dy
            if np.max(np.abs(F_ode(y_new, g, x_in, p_in))) < res:
                y = y_new; break
    return y, np.max(np.abs(F_ode(y, g, x_in, p_in)))

# ---------------------------------------------------------------------------
# Scattering matrix (full dense, 2N × 2N)
# ---------------------------------------------------------------------------
def scattering_matrix(Jac_sp):
    """
    S = I + K_sqrt @ Jac⁻¹ @ K_sqrt
    For κ = κ₁ = 1:  K_sqrt = I  →  S = I + Jac⁻¹

    Computes the full 2N x 2N dense S by factorising the sparse Jac once
    and solving for all 2N standard basis vectors.
    """
    solve = scipy.sparse.linalg.factorized(Jac_sp.tocsc())
    n2 = 2 * N
    # Jac⁻¹ = solve applied to each column of I
    # Note: factorized() returns a real solver → S is real at g=0
    G = np.zeros((n2, n2), dtype=np.float64)
    for j in range(n2):
        e_j = np.zeros(n2); e_j[j] = 1.0
        G[:, j] = solve(e_j)
    return np.eye(n2) + G          # S = I + Jac⁻¹

# ---------------------------------------------------------------------------
# U operators (real, 2N x 2N)
# ---------------------------------------------------------------------------
# Both are involutory: U² = I → U⁻¹ = U
I_N  = np.eye(N)
Z_N  = np.zeros((N, N))

U_x  = np.block([[Z_N,  I_N],   # σ_y (complex) / σ_x (quadrature): swap x↔p
                  [I_N,  Z_N]])  # imaginary pert δa=-iβerr ↔ p-input perturbation

U_z  = np.block([[I_N,   Z_N],  # σ_x (complex) / σ_z (quadrature): flip p→-p
                  [Z_N,  -I_N]]) # real pert δa=βerr ↔ x-input perturbation

# ---------------------------------------------------------------------------
# Quasi-symmetry angle
# ---------------------------------------------------------------------------
def quasi_sym_angle(S, U):
    """
    Frobenius angle between S† and USU⁻¹ (= USU since U²=I for our operators).

    Definition:
        ⟨A, B⟩_F  =  tr(A† B)
        cos θ(S†, USU) = Re⟨S†, USU⟩_F / (‖S†‖_F ‖USU‖_F)
                       = Re tr((S†)† USU) / ‖S‖²_F
                       = Re tr(S · USU) / ‖S‖²_F

    For real S: tr(S · USU) is the correct inner product.
    Note: tr(Sᵀ · USU) ≠ tr(S · USU) unless S is symmetric.

    Sanity check: at g=0 for any J, U_x Jac U_x = Jac^T
    → U_x S U_x = Sᵀ = S†  →  USU = S†  →  cos θ = 1  →  θ = 0°. ✓
    """
    USU    = U @ S @ U                        # U⁻¹ = U for both our operators
    inner  = np.real(np.trace(S @ USU))       # Re tr(S · USU)  [NOT S.T @ USU]
    norm2  = np.linalg.norm(S, 'fro') ** 2
    cos_val = np.clip(inner / norm2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val))), float(cos_val)

# ---------------------------------------------------------------------------
# Main loop 
# ---------------------------------------------------------------------------
print(f"\nComputing quasi-symmetry angles for {N_IMAGES} images x {len(G_VALUES)} g values\n")

shape     = (len(G_VALUES), N_IMAGES)
ang_x_all = np.full(shape, np.nan)   # σ_y complex / σ_x quadrature
ang_z_all = np.full(shape, np.nan)   # σ_x complex / σ_z quadrature
cos_x_all = np.full(shape, np.nan)
cos_z_all = np.full(shape, np.nan)
chi_ss_all= np.full(shape, np.nan)

for si in range(N_IMAGES):
    img    = test_imgs[si].astype(np.float64)
    x_in   = np.zeros(N);  x_in[:INPUT_PIXELS] = img.flatten() / 100.0
    y0     = analytic_ss_g0(x_in)

    print(f"  image {si}", end='', flush=True)

    for gi, g in enumerate(G_VALUES):
        # Find SS
        sol, res = newton_ss(y0, g, x_in)
        if res > 1e-7:
            print(f" [g={g:.1f}: SS failed, res={res:.1e}]", end='')
            continue
        y0 = sol

        x_ss, p_ss = sol[:N], sol[N:]
        phot  = np.mean((x_ss**2 + p_ss**2) / 2.0)
        chi   = g * phot / KAPPA_EFF
        chi_ss_all[gi, si] = chi

        # Build Jacobian and full S
        Jac_sp = jac_F_sparse(x_ss, p_ss, g)
        S      = scattering_matrix(Jac_sp)

        # Quasi-symmetry angles
        ang_x, cos_x = quasi_sym_angle(S, U_x)
        ang_z, cos_z = quasi_sym_angle(S, U_z)
        ang_x_all[gi, si] = ang_x
        ang_z_all[gi, si] = ang_z
        cos_x_all[gi, si] = cos_x
        cos_z_all[gi, si] = cos_z

        print(f"  g={g:.1f}:χ={chi:.3f}  θ(σ_y)={ang_x:.2f}°  θ(σ_x)={ang_z:.2f}°",
              end='', flush=True)

    print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print(f"{'g':>5}  {'χ_ss':>8}  "
      f"{'θ(σ_y complex) [°]':>22}  {'θ(σ_x complex) [°]':>22}")
print(f"       {'(= σ_x quad)':>8}  "
      f"{'mean ± std':>22}  {'mean ± std':>22}")
print("=" * 80)
for gi, g in enumerate(G_VALUES):
    chi_m = np.nanmean(chi_ss_all[gi])
    ax_m  = np.nanmean(ang_x_all[gi]);  ax_s = np.nanstd(ang_x_all[gi])
    az_m  = np.nanmean(ang_z_all[gi]);  az_s = np.nanstd(ang_z_all[gi])
    print(f"{g:5.1f}  {chi_m:8.4f}  "
          f"{ax_m:12.3f} ± {ax_s:.3f}     "
          f"{az_m:12.3f} ± {az_s:.3f}")
print()
print("θ = 0°  → perfect quasi-symmetry (SB gradient exact)")
print("θ = 90° → no quasi-symmetry")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
out_npz = os.path.join(SCRIPT_DIR, 'quasi_symmetry_results.npz')
np.savez(out_npz,
         G_VALUES    = G_VALUES,
         chi_ss_all  = chi_ss_all,
         ang_x_all   = ang_x_all,   # σ_y complex / σ_x quadrature
         ang_z_all   = ang_z_all,   # σ_x complex / σ_z quadrature
         cos_x_all   = cos_x_all,
         cos_z_all   = cos_z_all)
print(f"\nSaved  {out_npz}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
chi_means = [np.nanmean(chi_ss_all[gi]) for gi in range(len(G_VALUES))]

# ---------------------------------------------------------------------------
# Figure 1: angle vs chi_ss, mean ± std, both U operators side-by-side 
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, ang_all, label, short, color in [
        (axes[0], ang_x_all,
         r'$\sigma_y$ (complex) $=$ $\sigma_x$ (quad.)' + '\n'
         + r'imaginary pert. $\delta a = -i\beta e$',
         r'$\sigma_y$ complex', 'steelblue'),
        (axes[1], ang_z_all,
         r'$\sigma_x$ (complex) $=$ $\sigma_z$ (quad.)' + '\n'
         + r'real pert. $\delta a = \beta e$',
         r'$\sigma_x$ complex', 'tomato')]:

    ang_m = np.array([np.nanmean(ang_all[gi]) for gi in range(len(G_VALUES))])
    ang_s = np.array([np.nanstd(ang_all[gi])  for gi in range(len(G_VALUES))])

    # per-image scatter (light)
    for gi in range(len(G_VALUES)):
        vals = ang_all[gi][np.isfinite(ang_all[gi])]
        ax.scatter(np.full(len(vals), chi_means[gi]), vals,
                   s=18, color=color, alpha=0.25, zorder=2)

    # mean ± std
    ax.errorbar(chi_means, ang_m, yerr=ang_s, fmt='o-', color=color,
                lw=2.2, ms=8, capsize=5, zorder=5, label='mean ± std')

    # annotate g values
    for gi, g in enumerate(G_VALUES):
        if np.isfinite(ang_m[gi]):
            ax.annotate(f'g={g}', (chi_means[gi], ang_m[gi]),
                        textcoords='offset points', xytext=(6, 3), fontsize=8.5,
                        color=color)

    ax.axhline(0,  color='k',  ls='--', lw=1.0, alpha=0.6,
               label=r'0° — exact SB ($g=0$)')
    ax.axhline(90, color='grey', ls=':',  lw=1.0, alpha=0.5, label='90° (random)')
    ax.set_xlabel(r'$\chi_{\rm ss} = g\langle|a_{\rm ss}|^2\rangle/\kappa_{\rm eff}$',
                  fontsize=12)
    ax.set_ylabel(r'$\theta(U) = \angle(S^\dagger,\;USU^{-1})$  [°]', fontsize=11)
    ax.set_title(f'U = {label}', fontsize=10)
    ax.set_ylim(-1, max(np.nanmax(ang_z_all)*1.3, 20))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

fig.suptitle(
    f'MNIST Kerr network  (N={N}, κ=κ₁=1, self-Kerr G=I, seed 41, {N_IMAGES} images)\n'
    r'Quasi-symmetry angle of scattering matrix: $\theta(U)=\angle(S^\dagger,\,USU^{-1})$',
    fontsize=11)
fig.tight_layout()
out1 = os.path.join(SCRIPT_DIR, 'quasi_symmetry_angle.png')
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved  {out1}")

# ---------------------------------------------------------------------------
# Figure 2: both U on same axes — direct comparison
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7, 5))

for ang_all, label, color, marker in [
        (ang_x_all, r'$U=\sigma_y$ (complex) — imaginary pert.', 'steelblue', 'o'),
        (ang_z_all, r'$U=\sigma_x$ (complex) — real pert.',      'tomato',    's')]:
    ang_m = np.array([np.nanmean(ang_all[gi]) for gi in range(len(G_VALUES))])
    ang_s = np.array([np.nanstd(ang_all[gi])  for gi in range(len(G_VALUES))])
    ax2.errorbar(chi_means, ang_m, yerr=ang_s,
                 fmt=f'{marker}-', color=color, lw=2, ms=8, capsize=5,
                 label=label)

ax2.axhline(0,  color='k', ls='--', lw=1, alpha=0.6, label=r'$\theta=0°$ (exact)')
ax2.set_xlabel(r'$\chi_{\rm ss} = g\langle|a_{\rm ss}|^2\rangle/\kappa_{\rm eff}$',
               fontsize=12)
ax2.set_ylabel(r'$\theta(U)$  [°]', fontsize=12)
ax2.set_title(
    f'MNIST  (N={N}, self-Kerr, {N_IMAGES} images)\n'
    r'$\sigma_y$ vs $\sigma_x$ SB quasi-symmetry angle',
    fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
fig2.tight_layout()
out2 = os.path.join(SCRIPT_DIR, 'quasi_symmetry_comparison.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Saved  {out2}")
