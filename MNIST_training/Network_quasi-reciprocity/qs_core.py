"""
qs_core.py

Shared physics library for quasi-symmetry analysis of MNIST Kerr networks.
All functions are parametrised — no module-level globals — so multiple J
matrices can be used in the same session.
"""

import types
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


def make_system(J_np, KAPPA=1.0, KAPPA1=1.0):
    """
    Build all physics functions for a given connectivity matrix J_np.

    Returns a SimpleNamespace whose methods (jac_F_sparse, analytic_ss_g0,
    F_ode, newton_ss) are closures over J_np and kappa vectors, mirroring
    the structure of quasi_symmetry_angle.py but without global state.
    """
    N         = J_np.shape[0]
    kappa_v   = KAPPA  * np.ones(N)
    kappa1_v  = KAPPA1 * np.ones(N)
    sqrt_k    = np.sqrt(kappa_v)
    KAPPA_EFF = 0.5 * (KAPPA + KAPPA1)
    J_sp      = scipy.sparse.csr_matrix(J_np)

    # g=0 analytic SS: (I + J²) x = -x_in,  p = -J x  (valid for kappa=kappa1=1)
    A_g0 = np.eye(N) + J_np @ J_np
    L_g0 = scipy.linalg.lu_factor(A_g0)

    def jac_F_sparse(x_ss, p_ss, g):
        keff = 0.5 * (kappa_v + kappa1_v)
        d_xx = -keff + g * x_ss * p_ss
        d_xp =  0.5 * g * (x_ss**2 + 3 * p_ss**2)
        d_px = -0.5 * g * (3 * x_ss**2 + p_ss**2)
        d_pp = -keff - g * x_ss * p_ss
        return scipy.sparse.bmat([
            [scipy.sparse.diags(d_xx),          J_sp + scipy.sparse.diags(d_xp)],
            [-J_sp + scipy.sparse.diags(d_px),  scipy.sparse.diags(d_pp)       ]
        ], format='csc')

    def analytic_ss_g0(x_in):
        x = -scipy.linalg.lu_solve(L_g0, x_in)
        p = -J_np @ x
        return np.concatenate([x, p])

    def F_ode(y, g, x_in, p_in=None):
        x, p = y[:N], y[N:]
        if p_in is None:
            p_in = np.zeros(N)
        r  = x**2 + p**2
        Fx = -0.5*(kappa_v+kappa1_v)*x + 0.5*g*r*p + J_np@p - sqrt_k*x_in
        Fp = -0.5*(kappa_v+kappa1_v)*p - 0.5*g*r*x - J_np@x - sqrt_k*p_in
        return np.concatenate([Fx, Fp])

    def newton_ss(y0, g, x_in, tol=1e-9, maxiter=30):
        y    = y0.copy()
        p_in = np.zeros(N)
        for _ in range(maxiter):
            F   = F_ode(y, g, x_in, p_in)
            res = np.max(np.abs(F))
            if res < tol:
                break
            Jac = jac_F_sparse(y[:N], y[N:], g)
            dy  = scipy.sparse.linalg.spsolve(Jac, -F)
            for alpha in [1.0, 0.5, 0.25, 0.125]:
                y_new = y + alpha * dy
                if np.max(np.abs(F_ode(y_new, g, x_in, p_in))) < res:
                    y = y_new
                    break
        return y, np.max(np.abs(F_ode(y, g, x_in, p_in)))

    return types.SimpleNamespace(
        N             = N,
        KAPPA         = KAPPA,
        KAPPA1        = KAPPA1,
        KAPPA_EFF     = KAPPA_EFF,
        kappa_v       = kappa_v,
        kappa1_v      = kappa1_v,
        jac_F_sparse  = jac_F_sparse,
        analytic_ss_g0= analytic_ss_g0,
        F_ode         = F_ode,
        newton_ss     = newton_ss,
    )


def scattering_matrix(Jac_sp, N):
    """Full dense 2N×2N scattering matrix S = I + Jac⁻¹."""
    solve = scipy.sparse.linalg.factorized(Jac_sp.tocsc())
    n2    = 2 * N
    G     = np.zeros((n2, n2), dtype=np.float64)
    for j in range(n2):
        e_j     = np.zeros(n2)
        e_j[j]  = 1.0
        G[:, j] = solve(e_j)
    return np.eye(n2) + G


def quasi_sym_angle(S, U):
    """
    Frobenius angle between S and USU (quasi-symmetry condition: S† = USU⁻¹).

    cos θ = Re tr(S · USU) / ‖S‖²_F

    Returns (angle_degrees, cosine).
    """
    USU     = U @ S @ U
    inner   = np.real(np.trace(S @ USU))
    norm2   = np.linalg.norm(S, 'fro') ** 2
    cos_val = np.clip(inner / norm2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val))), float(cos_val)


def make_U_operators(N):
    """
    Return (U_x, U_z) — the two SB perturbation operators (real, 2N×2N).

    U_x = [[0, I], [I, 0]]   sigma_y complex / imaginary perturbation
    U_z = [[I, 0], [0, -I]]  sigma_x complex / real perturbation
    Both satisfy U² = I, so U⁻¹ = U.
    """
    I_N = np.eye(N)
    Z_N = np.zeros((N, N))
    U_x = np.block([[Z_N,  I_N], [I_N,  Z_N]])
    U_z = np.block([[I_N,  Z_N], [Z_N, -I_N]])
    return U_x, U_z
