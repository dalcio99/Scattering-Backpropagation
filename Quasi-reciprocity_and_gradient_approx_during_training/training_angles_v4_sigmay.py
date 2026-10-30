"""
training_angles_v4_sigmay.py  —  sigma_y-TRAINED VARIANT of training_angles_v4.py

Identical to training_angles_v4.py in every parameter (same seed list from
np.random.seed(777), same β, lr via SB_LR, epochs, hard-stop rules, warm-start
logic, diagnostics) EXCEPT that the weight update of the MAIN run uses the
SB_y rule (p-quadrature perturbation, code sigma_x branch):  J += (lr/β)·A_x.
Purpose: regenerate training_v4_fig1_sigmay with a run genuinely trained with
∇^{SB_y}C (the original main run trained with the SB_x rule and only monitored
the sigma_y quantities).  The method-comparison block is skipped by default
(SB_RUN_CMP=1 re-enables it); cmp_loss is saved as NaN so plot_v4.py still runs.
Output: results_v4_sigmay_lr<lr>/ — run plot_v4.py on it for the paper figure.

Original header follows.
──────────────────────────────────────────────────────────────────────────────
training_angles_v4.py  —  simplified, paper-oriented version

Distilled from training_angles_v3.py.  Removes the pattern-deactivation and
multistability machinery; adds a clean hard-stop criterion and diagnostics.

WHAT THIS SCRIPT DOES
---------------------
  1. HARD STOP.  A run (one Kerr-type * g * seed) is terminated the first epoch
     in which, for ANY XOR pattern, either
        (a) a steady state cannot be found          (fsolve residual > SS_TOL), or
        (b) the Jacobian is singular                (rcond < SINGULAR_RCOND, i.e.
            det(Jac)→0 so the inverse / Green's function does not exist).
     The stopping epoch and reason are recorded; all later epochs are NaN.

  2. DATASET.  Canonical XOR  [0,0],[1,0],[0,1],[1,1].

  3. LINEAR BASELINE.  g = 0 is included as a comparison condition: the cavity
     is then linear, so it CANNOT represent XOR and the loss plateaus at a finite
     floor — a reference against which the nonlinear (g>0) runs are read.

  4. NONLINEARITY DIAGNOSTICS, tracked per epoch and averaged over N nodes AND
     over the (solved) dataset patterns.  With a = (x + i p)/√2 so
     |a_j|² = (x_j² + p_j²)/2, and κ̄ = mean(κ):
        chi_ss = g · ⟨|ā|²⟩_avg / κ̄        (self-consistent, steady-state based)
        chi_in = g · ⟨|a_in|²⟩_avg / κ̄²    (input based; ≈ g by construction)

  5. S-MATRIX QUASI-SYMMETRY.  cos(S†, U S U⁻¹) for U = sigma_x and U = sigma_y
     (paper/a-basis).  = 1 when S obeys the corresponding reciprocity relation.

  6. GRADIENT ALIGNMENT.  cos(∇_exact, ∇_SB^x) and cos(∇_exact, ∇_SB^y) along the
     training trajectory (∇_exact from the implicit-function theorem / Jacobian
     inverse).

  7. METHOD COMPARISON — TRAINING WITH DIFFERENT GRADIENTS.  From the SAME J init
     we independently train three networks and compare their LOSS trajectories:
        sigma_x SB   (x-quadrature perturbation)
        sigma_y SB   (p-quadrature perturbation)
        exact    (IFT / Jacobian-inverse gradient)
     Each obeys the same hard-stop rule.

  Output folder: results_v4/.

──────────────────────────────────────────────────────────────────────────────
NOTATION  (IMPORTANT — code basis vs paper basis)
──────────────────────────────────────────────────────────────────────────────
Code works in the (x, p) quadrature basis; the paper works in the complex
amplitude (a) basis, a = (x + i p)/√2.  The feedback symmetry operators map:

    code  sigma_z  (x-quadrature perturbation of a_in)  ⟷  paper  sigma_x   → "SB_x"
    code  sigma_x  (p-quadrature perturbation of a_in)  ⟷  paper  sigma_y   → "SB_y"

Internal variables keep the code-basis names (sb_z_vec, sigma_z, ...), but every
PLOT LABEL uses the paper basis (sigma_x, sigma_y, ∇_SB^x, ∇_SB^y).  The weight update
that actually trains the main run is the SB_x rule (x-quadrature perturbation).
Both SB gradients are exact at g = 0.

Loss convention: with label_rescaling = 10, err = y - target/10 and the plotted
loss is label_rescaling²·err² = (10·y - target)²  (loss on the rescaled output).
"""

import os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import fsolve
import jax, jax.numpy as jnp
from jax import jacfwd
warnings.filterwarnings('ignore')
jax.config.update('jax_enable_x64', True)
print('JAX devices:', jax.devices())

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Nearest-neighbour Kerr topology: ring (periodic, default) or line (open boundary).
_NN_LINE = os.environ.get('SB_NN_LINE', '0') == '1'

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N          = 10
kappa      = np.ones(N)
kbar       = float(np.mean(kappa))          # κ̄
out_idx    = N // 2
beta       = 0.01
lr         = float(os.environ.get('SB_LR', 0.1))   # override with SB_LR=0.05 …
num_epochs = int(os.environ.get('SB_EPOCHS', 1000))
# output dir: canonical results_v4/ for lr=0.1, otherwise results_v4_lr<lr>/;
# append _nnline when the nearest-neighbour Kerr uses an open (line) boundary.
_base = 'results_v4_sigmay' if abs(lr-0.1) < 1e-12 else f"results_v4_sigmay_lr{lr:g}"
OUT = os.path.join(_SCRIPT_DIR, _base + ('_nnline' if _NN_LINE else ''))
_RUN_CMP = os.environ.get('SB_RUN_CMP', '0') == '1'   # method comparison off by default
os.makedirs(OUT, exist_ok=True)
num_seeds  = 5

SS_TOL         = 1e-6      # steady state "found" if max|F| < SS_TOL
SINGULAR_RCOND = 1e-12     # Jacobian "singular" (det→0) if 1/cond(Jac) < this

I_N = np.eye(N);  O_N = np.zeros((N, N))
sigma_z = np.block([[I_N, O_N], [O_N, -I_N]])   # code σ_z  ↔  paper σ_x
sigma_x = np.block([[O_N, I_N], [I_N,  O_N]])   # code σ_x  ↔  paper σ_y
k_sqrt  = np.diag(np.tile(np.sqrt(kappa), 2))

XOR_inputs  = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
XOR_targets = np.array([0, 1, 1, 0])
label_rescaling = 10.0
n_inputs = len(XOR_inputs)

# ---------------------------------------------------------------------------
# G matrices
# ---------------------------------------------------------------------------
def build_G_self(N):      
    return np.eye(N)
def build_G_cross_nn(N):
    """Nearest-neighbour cross-Kerr.  Ring (periodic) by default; open line if
    SB_NN_LINE=1 (nodes 0 and N-1 keep only one neighbour, no wrap-around)."""
    G = np.zeros((N, N))
    for i in range(N):
        for j in (i-1, i+1):
            if _NN_LINE:
                if 0 <= j < N: G[i, j] = 1
            else:
                G[i, j % N] = 1
    return G
def build_G_all_to_all(N): return np.ones((N, N))

# ---------------------------------------------------------------------------
# cosine-similarity helpers
# ---------------------------------------------------------------------------
def frobenius_cosine(A, B):
    """Hermitian Frobenius cosine  Re⟨A,B⟩_F / (‖A‖_F ‖B‖_F),  ⟨A,B⟩=tr(A† B)."""
    ip = np.real(np.vdot(A, B))            # vdot conjugates its first argument
    nA, nB = np.linalg.norm(A), np.linalg.norm(B)
    if nA < 1e-14 or nB < 1e-14: return np.nan
    return float(np.clip(ip/(nA*nB), -1., 1.))

def quasi_sym_cos(S, U):
    """cos(S†, U S U⁻¹) — quasi-symmetry / reciprocity of the scattering matrix."""
    return frobenius_cosine(S.conj().T, U @ S @ np.linalg.inv(U))

def vec_cosine(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-14 or nv < 1e-14: return np.nan
    return float(np.clip(np.dot(u,v)/(nu*nv), -1., 1.))

# ---------------------------------------------------------------------------
# equilibrium solver 
# ---------------------------------------------------------------------------

# Physical field amplitude scale: |a_in|² ~ N ⇒ quadratures ~ √(2N).
_AMP = np.sqrt(2.0 * N)
N_RAND_STARTS = 10          # restarts scaled to the physical amplitude

def _ss_g0(J, x_in, p_in):
    """Analytic g=0 (linear) steady state — a good physical-scale warm start."""
    M = np.block([[-0.5*np.diag(kappa), J], [-J, -0.5*np.diag(kappa)]])
    b = np.concatenate([np.sqrt(kappa)*x_in, np.sqrt(kappa)*p_in])
    try:
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return np.zeros(2*N)

def find_eq(J, G, g, x_in, p_in, y0=None):
    """Find a steady state F(y)=0.  Returns (sol, residual).

    Starts are tried in order of physical preference — warm start (branch
    continuity), analytic g=0 solution, zeros — then several restarts scaled to
    the physical amplitude √(2N).  The first converged start is accepted, so the
    warm-connected branch is preferred; the random restarts only serve to avoid
    a spurious "no steady state" when the solver merely missed from a poor start.
    residual > SS_TOL  ⇒  genuinely no steady state found."""
    def F(y):
        x, p = y[:N], y[N:]
        r  = G @ (x**2 + p**2)
        Fx = -0.5*kappa*x + J@p + 0.5*g*p*r - np.sqrt(kappa)*x_in
        Fp = -0.5*kappa*p - J@x - 0.5*g*x*r - np.sqrt(kappa)*p_in
        return np.concatenate([Fx, Fp])
    starts = (([y0] if y0 is not None else [])
              + [_ss_g0(J, x_in, p_in), np.zeros(2*N)]
              + [_AMP * np.random.randn(2*N) for _ in range(N_RAND_STARTS)])
    best_res, best_sol = np.inf, np.zeros(2*N)
    for s in starts:
        sol, info, ier, _ = fsolve(F, s, full_output=True)
        res = np.max(np.abs(info['fvec']))
        if res < best_res:
            best_res, best_sol = res, sol
        if ier == 1 and res < SS_TOL:
            break
    return best_sol, best_res

def jac_singular(Jac):
    """True if the Jacobian is (numerically) non-invertible → no Green's function."""
    rcond = 1.0 / np.linalg.cond(Jac)
    return (not np.isfinite(rcond)) or (rcond < SINGULAR_RCOND), rcond

# ---------------------------------------------------------------------------
# JAX Jacobian 
# ---------------------------------------------------------------------------
def make_jac_fn(G_arr, g_val):
    G_j = jnp.array(G_arr);  k_j = jnp.array(kappa);  g_ = float(g_val)
    @jax.jit
    def jac_fn(y, J_arr, x_in_arr, p_in_arr):
        def F(y_):
            x, p = y_[:N], y_[N:]
            r  = G_j @ (x**2 + p**2)
            Fx = -0.5*k_j*x + J_arr@p + 0.5*g_*p*r - jnp.sqrt(k_j)*x_in_arr
            Fp = -0.5*k_j*p - J_arr@x - 0.5*g_*x*r - jnp.sqrt(k_j)*p_in_arr
            return jnp.concatenate([Fx, Fp])
        return jacfwd(F)(y)
    return jac_fn

# ---------------------------------------------------------------------------
#  vectorized dF/dJ outer product (symmetric matrix J, diagonal halved) 
# ---------------------------------------------------------------------------
def _outer_sym(a, b):
    M = np.outer(a, b) + np.outer(b, a)
    M[np.diag_indices(N)] *= 0.5
    return M

def compute_grad_matrices(x_free, p_free, inv_Jac, err, sigz_dy, sigx_dy):
    """Upper-triangle gradient vectors:
         true_g_vec : exact (implicit-function-theorem) gradient
         sb_z_vec   : SB_x gradient (x-quadrature perturbation, code sigma_z branch)
         sb_x_vec   : SB_y gradient (p-quadrature perturbation, code sigma_x branch)
       Also returns A_z (weight-update matrix for the SB_x rule) and A_true."""
    upper_tri = [(j, l) for j in range(N) for l in range(j, N)]
    row_jac = -inv_Jac[out_idx, :]
    rx, rp   = row_jac[:N], row_jac[N:]
    sx_z, sp_z = sigz_dy[:N], sigz_dy[N:]
    sx_x, sp_x = sigx_dy[:N], sigx_dy[N:]

    A_true = _outer_sym(rx,   p_free) - _outer_sym(rp,   x_free)
    A_z    = _outer_sym(sx_z, p_free) - _outer_sym(sp_z, x_free)
    A_x    = _outer_sym(sx_x, p_free) - _outer_sym(sp_x, x_free)

    idx = ([j for j,l in upper_tri], [l for j,l in upper_tri])
    true_g_vec = 2.0 * err   * A_true[idx]
    sb_z_vec   = -(0.5/beta) * A_z   [idx]
    sb_x_vec   = -(0.5/beta) * A_x   [idx]
    return true_g_vec, sb_z_vec, sb_x_vec, A_z, A_x, A_true

# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------
g_values   = [0.0, 0.05, 0.2, 0.5, 1.0, 3., 5.]  # g=0 → linear baseline
kerr_types = [
    ("Self-Kerr",               build_G_self),
    ("Cross-Kerr (n.n.)",       build_G_cross_nn),
    ("Cross-Kerr (all-to-all)", build_G_all_to_all),
]
nK = len(kerr_types)
nG = len(g_values)
n_pairs = N * (N + 1) // 2

np.random.seed(777)
seeds = [np.random.randint(0, 99999) for _ in range(num_seeds)]
print(f"Seeds: {seeds}")
print(f"Epochs: {num_epochs}  |  Seeds: {num_seeds}  |  beta={beta}  lr={lr}")
print(f"XOR inputs: {XOR_inputs.tolist()}   g_values (0=linear): {g_values}")
print(f"Stop if: residual>{SS_TOL} (no SS)  OR  rcond(Jac)<{SINGULAR_RCOND} (singular)\n")

# ---------------------------------------------------------------------------
# storage 
# ---------------------------------------------------------------------------
shape = (nK, nG, num_seeds, num_epochs)
all_loss   = np.full(shape, np.nan)
all_cosSx  = np.full(shape, np.nan)    # cos(S†, σ_x S σ_x⁻¹)  [a-basis σ_x = code σ_z]
all_cosSy  = np.full(shape, np.nan)    # cos(S†, σ_y S σ_y⁻¹)  [a-basis σ_y = code σ_x]
all_cosGz  = np.full(shape, np.nan)    # cos(∇_exact, ∇_SB^x)  [x-quad pert]
all_cosGx  = np.full(shape, np.nan)    # cos(∇_exact, ∇_SB^y)  [p-quad pert]
all_chi_ss = np.full(shape, np.nan)    # g⟨|ā|²⟩/κ̄  (dataset+node avg)
all_chi_in = np.full(shape, np.nan)    # g⟨|a_in|²⟩/κ̄²
all_invjac_norm = np.full(shape, np.nan)
all_stop_epoch  = np.full((nK, nG, num_seeds), num_epochs, dtype=int)   # num_epochs = never stopped
all_stop_reason = np.empty((nK, nG, num_seeds), dtype=object)

UPDATE_RULE_LABEL = r"$\sigma_y$ SB update (p-quadrature pert.)"

# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------
for ki, (kname, G_fn) in enumerate(kerr_types):
    G = G_fn(N)
    for gi, g in enumerate(g_values):
        print(f"\n{'='*55}\n{kname},  g = {g:.2f}" + ("  (LINEAR)" if g == 0 else "") + f"\n{'='*55}")
        jac_fn = make_jac_fn(G, g)

        for si, seed in enumerate(seeds):
            np.random.seed(seed)
            sigma = 1.0 / np.sqrt(N)
            R = sigma * np.random.randn(N, N);  U = np.triu(R)
            J = U + U.T - np.diag(np.diag(U))
            y0 = np.zeros(2*N)
            stop_reason = None

            for epoch in range(num_epochs):
                epoch_loss = 0.0;  n_found = 0
                epoch_cosSx, epoch_cosSy = [], []
                epoch_cosGz, epoch_cosGx = [], []
                epoch_invjac = []
                chi_ss_vals, chi_in_vals = [], []

                for inp_idx, (inp, tgt) in enumerate(zip(XOR_inputs, XOR_targets)):
                    x_in = np.zeros(N);  x_in[:2] = np.sqrt(2*N) * inp.astype(float)
                    p_in = np.zeros(N)

                    # free steady state
                    sol_f, res_f = find_eq(J, G, g, x_in, p_in, y0)
                    if res_f > SS_TOL:
                        stop_reason = f"no steady state (pattern {inp_idx}={inp.tolist()})"
                        break
                    y0 = sol_f
                    x_free, p_free = sol_f[:N], sol_f[N:]
                    err = x_free[out_idx] - tgt / label_rescaling

                    # Jacobian: require invertibility (Green's function nonsingular)
                    Jac = np.array(jac_fn(jnp.array(sol_f), jnp.array(J),
                                          jnp.array(x_in), jnp.array(p_in)))
                    sing, rcond = jac_singular(Jac)
                    if sing:
                        stop_reason = f"singular Jacobian (pattern {inp_idx}, rcond={rcond:.1e})"
                        break
                    inv_Jac = np.linalg.inv(Jac)
                    S = np.eye(2*N) + k_sqrt @ inv_Jac @ k_sqrt

                    # nonlinearity diagnostics (a = (x+ip)/√2)
                    mean_phot_ss = np.mean(x_free**2 + p_free**2) / 2.0   # ⟨|ā|²⟩ node-avg
                    mean_phot_in = np.mean(x_in**2  + p_in**2)   / 2.0    # ⟨|a_in|²⟩ node-avg
                    chi_ss_vals.append(g * mean_phot_ss / kbar)
                    chi_in_vals.append(g * mean_phot_in / kbar**2)
                    epoch_invjac.append(np.linalg.norm(inv_Jac, 'fro'))

                    # S quasi-symmetry
                    epoch_cosSx.append(quasi_sym_cos(S, sigma_z))   # a-basis σ_x
                    epoch_cosSy.append(quasi_sym_cos(S, sigma_x))   # a-basis σ_y

                    # SB steady states (x-quad and p-quad perturbations
                    x_in_p = x_in.copy();  x_in_p[out_idx] += beta * 2.0 * err
                    sol_px, res_px = find_eq(J, G, g, x_in_p, p_in, y0)
                    if res_px > SS_TOL:
                        stop_reason = f"no steady state (x-pert, pattern {inp_idx})"
                        break

                    p_in_p = p_in.copy();  p_in_p[out_idx] += beta * 2.0 * err
                    sol_pp, res_pp = find_eq(J, G, g, x_in, p_in_p, sol_f)
                    if res_pp > SS_TOL:
                        stop_reason = f"no steady state (p-pert, pattern {inp_idx})"
                        break

                    y0 = sol_pp
                    n_found += 1
                    epoch_loss += label_rescaling**2 * err**2

                    sigz_dy = sigma_z @ (sol_px - sol_f)
                    sigx_dy = sigma_x @ (sol_pp - sol_f)
                    true_g_vec, sb_z_vec, sb_x_vec, A_z, A_x, _ = compute_grad_matrices(
                        x_free, p_free, inv_Jac, err, sigz_dy, sigx_dy)

                    epoch_cosGz.append(vec_cosine(true_g_vec, sb_z_vec))  # ∇_SB^x
                    epoch_cosGx.append(vec_cosine(true_g_vec, sb_x_vec))  # ∇_SB^y

                    # weight update: SB_y rule (p-quadrature perturbation)
                    J += (lr / beta) * A_x

                # end patterns
                if stop_reason is not None:
                    break   # HARD STOP: terminate this run

                all_loss  [ki,gi,si,epoch] = epoch_loss / n_found if n_found else np.nan
                all_cosSx [ki,gi,si,epoch] = np.nanmean(epoch_cosSx) if epoch_cosSx else np.nan
                all_cosSy [ki,gi,si,epoch] = np.nanmean(epoch_cosSy) if epoch_cosSy else np.nan
                all_cosGz [ki,gi,si,epoch] = np.nanmean(epoch_cosGz) if epoch_cosGz else np.nan
                all_cosGx [ki,gi,si,epoch] = np.nanmean(epoch_cosGx) if epoch_cosGx else np.nan
                all_chi_ss[ki,gi,si,epoch] = np.mean(chi_ss_vals) if chi_ss_vals else np.nan
                all_chi_in[ki,gi,si,epoch] = np.mean(chi_in_vals) if chi_in_vals else np.nan
                all_invjac_norm[ki,gi,si,epoch] = np.mean(epoch_invjac) if epoch_invjac else np.nan

            # end epochs
            if stop_reason is not None:
                all_stop_epoch [ki,gi,si] = epoch
                all_stop_reason[ki,gi,si] = stop_reason
                tag = f"STOPPED @ep{epoch}: {stop_reason}"
            else:
                all_stop_reason[ki,gi,si] = "completed"
                tag = "completed"
            last = num_epochs-1 if stop_reason is None else max(epoch-1, 0)
            print(f"  Seed {si+1}/{num_seeds}  {tag}  "
                  f"loss={all_loss[ki,gi,si,last]:.3e}  "
                  f"chi_ss={all_chi_ss[ki,gi,si,last]:.3f}  "
                  f"cos(exact,SBy)={all_cosGx[ki,gi,si,last]:.3f}")

print("\n✓  Main training complete.")

# -------------------------------------------------------------------------------
# METHOD COMPARISON — train with σ_x SB / σ_y SB / exact gradients (same J init)
# -------------------------------------------------------------------------------
#   'sz'  → σ_x SB (x-quadrature perturbation, code σ_z branch)
#   'sx'  → σ_y SB (p-quadrature perturbation, code σ_x branch)
#   'exact' → exact IFT gradient (Jacobian-inverse response)
METHOD_SHORT  = ['sz', 'sx', 'exact']
METHOD_NAMES  = [r'$\sigma_x$ SB', r'$\sigma_y$ SB', 'exact (IFT)']
METHOD_COLORS = ['steelblue', 'tomato', 'forestgreen']
nMcmp = len(METHOD_SHORT)

cmp_loss       = np.full((nMcmp, nK, nG, num_seeds, num_epochs), np.nan)
cmp_stop_epoch = np.full((nMcmp, nK, nG, num_seeds), num_epochs, dtype=int)

print("\n" + "="*60)
print("Method comparison: σ_x SB | σ_y SB | exact (IFT) — same J init"
      + ("" if _RUN_CMP else "   [SKIPPED — set SB_RUN_CMP=1 to enable]"))
print("="*60)

for ki, (kname, G_fn) in enumerate(kerr_types if _RUN_CMP else []):
    G = G_fn(N)
    for gi, g in enumerate(g_values):
        print(f"\n  {kname},  g = {g:.2f}")
        jac_fn_cmp = make_jac_fn(G, g)

        for si, seed in enumerate(seeds):
            np.random.seed(seed)
            s0 = 1.0/np.sqrt(N);  R0 = s0*np.random.randn(N,N);  U0 = np.triu(R0)
            J0 = U0 + U0.T - np.diag(np.diag(U0))

            for mi, method in enumerate(METHOD_SHORT):
                J = J0.copy();  y0_ = np.zeros(2*N);  stop = None

                for epoch in range(num_epochs):
                    ep_loss = 0.0;  n_f = 0
                    for inp_idx, (inp, tgt) in enumerate(zip(XOR_inputs, XOR_targets)):
                        x_in = np.zeros(N);  x_in[:2] = np.sqrt(2*N)*inp.astype(float)
                        p_in = np.zeros(N)

                        sol_f, res_f = find_eq(J, G, g, x_in, p_in, y0_)
                        if res_f > SS_TOL:
                            stop = f"no SS (pattern {inp_idx})"; break
                        y0_ = sol_f
                        x_free, p_free = sol_f[:N], sol_f[N:]
                        err = x_free[out_idx] - tgt/label_rescaling

                        Jac_ = np.array(jac_fn_cmp(jnp.array(sol_f), jnp.array(J),
                                                    jnp.array(x_in), jnp.array(p_in)))
                        sing, rcond = jac_singular(Jac_)
                        if sing:
                            stop = f"singular Jac (pattern {inp_idx})"; break
                        inv_Jac_ = np.linalg.inv(Jac_)

                        if method == 'sz':
                            x_in_p = x_in.copy();  x_in_p[out_idx] += beta*2*err
                            sol_p, res_p = find_eq(J, G, g, x_in_p, p_in, y0_)
                            if res_p > SS_TOL: stop = f"no SS x-pert (pat {inp_idx})"; break
                            y0_ = sol_p
                            dy = sigma_z @ (sol_p - sol_f)
                        elif method == 'sx':
                            p_in_p = p_in.copy();  p_in_p[out_idx] += beta*2*err
                            sol_p, res_p = find_eq(J, G, g, x_in, p_in_p, sol_f)
                            if res_p > SS_TOL: stop = f"no SS p-pert (pat {inp_idx})"; break
                            dy = sigma_x @ (sol_p - sol_f)
                        else:  # exact IFT response
                            delta_y_an = (inv_Jac_[:, out_idx]
                                          * np.sqrt(kappa[out_idx]) * beta * 2.0 * err)
                            dy = sigma_z @ delta_y_an

                        n_f += 1
                        ep_loss += label_rescaling**2 * err**2
                        sx_, sp_ = dy[:N], dy[N:]
                        A = _outer_sym(sx_, p_free) - _outer_sym(sp_, x_free)
                        J += (lr / beta) * A

                    if stop is not None:
                        break
                    cmp_loss[mi,ki,gi,si,epoch] = ep_loss/n_f if n_f else np.nan

                if stop is not None:
                    cmp_stop_epoch[mi,ki,gi,si] = epoch

            print(f"    Seed {si+1}  " + "  ".join(
                f"{nm}: {cmp_loss[mi,ki,gi,si, (num_epochs-1) if cmp_stop_epoch[mi,ki,gi,si]==num_epochs else max(cmp_stop_epoch[mi,ki,gi,si]-1,0)]:.2e}"
                for mi, nm in enumerate(METHOD_SHORT)))

print("\n✓  Method comparison complete.")

# ---------------------------------------------------------------------------
# save data
# ---------------------------------------------------------------------------
np.savez(os.path.join(OUT, 'training_angles_v4.npz'),
         loss=all_loss, cosSx=all_cosSx, cosSy=all_cosSy,
         cosGz=all_cosGz, cosGx=all_cosGx,
         chi_ss=all_chi_ss, chi_in=all_chi_in, invjac_norm=all_invjac_norm,
         stop_epoch=all_stop_epoch,
         cmp_loss=cmp_loss, cmp_stop_epoch=cmp_stop_epoch,
         seeds=np.array(seeds), g_values=np.array(g_values),
         beta=beta, lr=lr, N=N, kbar=kbar)
print("Saved  training_angles_v4.npz")

# ---------------------------------------------------------------------------
# Plotting  —  two main figures
# # ---------------------------------------------------------------------------
epochs   = np.arange(1, num_epochs + 1)
g_colors = plt.cm.viridis(np.linspace(0.05, 0.92, nG))
SEED_COLOR = '0.75'
LOSS_REFS  = [1e-4, 1e-6]

# Per-(kerr, g) nonlinearity numbers for the legend.
#   g/κ̄            : bare drive strength (κ̄=1 here so = g)
#   χ_ss(0)        : g⟨|ā|²⟩/κ̄ at the initial network (epoch 0) — evolves during
#                    training, so we quote the initial operating point
#   χ_in           : g⟨|a_in|²⟩/κ̄² — fixed; = g here since the √(2N) input scaling
#                    makes the dataset+node average ⟨|a_in|²⟩ = 1
chi_ss_init = np.nanmean(all_chi_ss[:, :, :, 0], axis=2)          # (nK, nG)
chi_in_rep  = np.nanmean(all_chi_in.reshape(nK, nG, -1), axis=2)  # (nK, nG) ≈ g

def legend_label(ki, gi):
    g = g_values[gi]
    gk = g / kbar
    return (f"{gk:.1f} , {chi_ss_init[ki,gi]:.2f} , {chi_in_rep[ki,gi]:.2f}"
            + ("   (linear)" if g == 0 else ""))

LEGEND_TITLE = (r"$g/\bar\kappa$ , $g\langle|\bar a|^2\rangle/\bar\kappa|_{t=0}$ , "
                r"$g\langle|a_{\rm in}|^2\rangle/\bar\kappa^2$")

# ---------------------------------------------------------------------------
# FIGURE 1 (one per symmetry)
# ---------------------------------------------------------------------------
#   3 columns (Self-Kerr, Cross-Kerr n.n., Cross-Kerr all-to-all)
#   3 rows: loss | cos(S†, U S U⁻¹) | cos(∇_exact, ∇_SB)   (U = σ_x or σ_y)
#   color = nonlinearity g/κ̄ 
def make_fig1(symm):
    if symm == 'x':
        cosS, cosG = all_cosSx, all_cosGz
        s_lab = r'$\cos(S^\dagger,\ \sigma_x S\,\sigma_x^{-1})$'
        g_lab = r'$\cos(\nabla_\theta\mathcal{C},\ \nabla_\theta^{{\rm SB}_x}\mathcal{C})$'
        title = r'$\sigma_x$ feedback (x-quadrature perturbation)'
    else:
        cosS, cosG = all_cosSy, all_cosGx
        s_lab = r'$\cos(S^\dagger,\ \sigma_y S\,\sigma_y^{-1})$'
        g_lab = r'$\cos(\nabla_\theta\mathcal{C},\ \nabla_\theta^{{\rm SB}_y}\mathcal{C})$'
        title = r'$\sigma_y$ feedback (p-quadrature perturbation)'

    rows = [(all_loss, 'Loss', True, None, None),
            (cosS, s_lab, False, -1.02, 1.02),
            (cosG, g_lab, False, -1.02, 1.02)]

    fig, axes = plt.subplots(3, nK, figsize=(5.3*nK, 3.1*3), sharey='row')
    for mi, (dat4d, mlabel, do_log, ymin, ymax) in enumerate(rows):
        for ki, (kname, _) in enumerate(kerr_types):
            ax = axes[mi, ki]
            for gi, g in enumerate(g_values):
                mean = np.nanmean(dat4d[ki, gi], axis=0)
                if do_log:
                    for si in range(num_seeds):
                        ax.plot(epochs, dat4d[ki,gi,si], color=g_colors[gi], lw=0.4, alpha=0.20)
                    ax.plot(epochs, mean, color=g_colors[gi], lw=2.0, label=legend_label(ki, gi))
                else:
                    std = np.nanstd(dat4d[ki, gi], axis=0)
                    ax.plot(epochs, mean, color=g_colors[gi], lw=2.0)
                    ax.fill_between(epochs, mean-std, mean+std, color=g_colors[gi], alpha=0.12)
            ax.set_xscale('log'); ax.set_xlim(epochs[0], epochs[-1]); ax.grid(alpha=0.25)
            if do_log:
                ax.set_yscale('log')
                for ref, ls in zip(LOSS_REFS, ['--', ':']):
                    ax.axhline(ref, color='k', lw=0.8, ls=ls, alpha=0.5)
                ax.set_title(kname, fontsize=11)
                ax.legend(fontsize=6.0, loc='upper right', title=LEGEND_TITLE,
                          title_fontsize=6.0, labelspacing=0.25, handlelength=1.3)
            else:
                ax.set_ylim(ymin, ymax)
                ax.axhline(1.0, color='k', lw=0.6, ls=':', alpha=0.5)
            if mi == 2: ax.set_xlabel("Epoch", fontsize=10)
            if ki == 0: ax.set_ylabel(mlabel, fontsize=11)
    fig.suptitle(f"XOR training — {title}   (mean over {num_seeds} seeds; "
                 r"color = nonlinearity $g/\bar\kappa$)", fontsize=12, y=1.003)
    fig.tight_layout()
    fn = f'training_v4_fig1_sigma{symm}.png'
    fig.savefig(os.path.join(OUT, fn), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved  {fn}")

make_fig1('x')
make_fig1('y')

# ----------------------------------------------
# FIGURE 2 — method comparison by nonlinearity 
# ----------------------------------------------
#   3 columns (Self / n.n. / all-to-all) × one row per g/κ̄ (0.2 … 5, no linear)
#   each panel: loss trained with ∇^{SB_x}, ∇^{SB_y}, and exact ∇ (same J init)
METHOD_PLOT_NAMES = [r'$\nabla_\theta^{{\rm SB}_x}\mathcal{C}$',
                     r'$\nabla_\theta^{{\rm SB}_y}\mathcal{C}$',
                     r'$\nabla_\theta\mathcal{C}$ (exact)']
g2_idx = [gi for gi, g in enumerate(g_values) if g > 0]
nR2 = len(g2_idx)
fig, axes = plt.subplots(nR2, nK, figsize=(4.4*nK, 2.7*nR2), sharex=True)
for r, gi in enumerate(g2_idx):
    g = g_values[gi]
    for ki, (kname, _) in enumerate(kerr_types):
        ax = axes[r, ki]
        for mi, (mname, mcol) in enumerate(zip(METHOD_PLOT_NAMES, METHOD_COLORS)):
            ax.plot(epochs, np.nanmean(cmp_loss[mi,ki,gi], axis=0),
                    color=mcol, lw=2.0, label=mname)   # mean only (log axis)
        for ref, ls in zip(LOSS_REFS, ['--', ':']):
            ax.axhline(ref, color='k', lw=0.8, ls=ls, alpha=0.45)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(epochs[0], epochs[-1]); ax.grid(alpha=0.25)
        if r == 0: ax.set_title(kname, fontsize=11)
        if r == nR2-1: ax.set_xlabel("Epoch", fontsize=10)
        if ki == 0: ax.set_ylabel(fr"$g/\bar\kappa={g:g}$" + "\nLoss", fontsize=10)
        if r == 0 and ki == nK-1: ax.legend(fontsize=8, loc='upper right')
fig.suptitle(r"XOR loss trained with $\nabla_\theta^{{\rm SB}_x}\mathcal{C}$, "
             r"$\nabla_\theta^{{\rm SB}_y}\mathcal{C}$ and exact $\nabla_\theta\mathcal{C}$"
             f"   (same $J$ init, mean over {num_seeds} seeds)", fontsize=12, y=1.002)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'training_v4_fig2_method.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved  training_v4_fig2_method.png")

# ----------------------------------------
#  stop-epoch summary table (printed) 
# ----------------------------------------
def _mean_stop(se_arr):
    stopped = se_arr[se_arr < num_epochs]
    return f"{np.mean(stopped)+1:.0f}" if len(stopped) else "none"

print("\nStop-epoch summary (mean over stopped seeds; 'none' = all completed):")
for ki, (kname, _) in enumerate(kerr_types):
    row = "  ".join(f"g/κ̄={g:g}: {_mean_stop(all_stop_epoch[ki,gi])}"
                    for gi, g in enumerate(g_values))
    print(f"  {kname:26s}  {row}")

print("\nAll outputs written to:", OUT)
