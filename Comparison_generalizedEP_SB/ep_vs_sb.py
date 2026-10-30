"""
ep_vs_sb.py — vector-field EP vs Scattering Backpropagation (deterministic).

Vactor field EP [1]: real state s = (x,p), nudge as a
force -beta grad_s C added to the dynamics, update nu = (d mu/d theta)^T (delta s)/beta
with P1 = P2 = identity. For a cost depending on the x-quadrature of the output
only, the EP force-nudge equals the SB sigma_x input nudge (delta a_in = +beta err),
so the two methods share the SAME free and nudged simulations and differ ONLY in
the contraction:

  SB (sigma_x):  G_{jl} ∝  Im[ abar_l  * delta_a_j ]    (unconjugated: transpose /
                                                          reciprocity pairing)
  EP:      G_{jl} ∝ -Im[ conj(abar_l) delta_a_j ] (conjugated: Hermitian
                                                          pairing)

Identity behind it: for the Bogoliubov Jacobian, M^dagger = sigma_x M sigma_x
<=> M^T = M, i.e. "EP with the transpose in the (a,a*) basis" IS SB with
U = sigma_x. The real-basis (P = 1) prescription is the only EP variant
that is not already SB.

Why there is no regime where vector field EP works on this platform: the real Jacobian
[[-k/2, J], [-J, -k/2]] is symmetric only at J = Delta = 0 — but there every
trainable parameter acts by rotating x into p (invisible in the x-readout), so
the gradient itself vanishes at first order. EP-exactness holds only where
there is nothing to learn: the mismatch is structural, not quantitative.

Panel (a): epoch-1 cosine vs true (IFT) gradient as a function of the effective
           nonlinearity (standard sweep setting, N = 10, XOR, noiseless).
Panel (b): g = 0 (SB exact by theorem), cosine vs coupling scale s (J -> s*J0),
           analytic steady states (no time stepping, no statistical error
           beyond seed-to-seed spread).

Validation: the simulated EP estimate is checked against its closed form at
g = 0 (the IFT adjoint solve with Jac in place of Jac^T) to cos = 1.

Run:  /opt/miniconda3/envs/jax-env/bin/python ep_vs_sb.py
(imports sc_core.py / gradient_worker_sc.py from the parent folder)

Citations:
[1] Scellier, Benjamin, et al. "Generalization of equilibrium propagation to vector 
                                field dynamics." arXiv preprint arXiv:1808.04873 (2018).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'Noise_analysis'))  # shared modules (sc_core, gradient_worker_sc)
from sc_core import simulate_record, analytic_ss_g0, XOR_INPUTS, XOR_TARGETS
from gradient_worker_sc import ift_gradient_batch, _make_J, _cosine

# ---------------------------------------------------------------------------
# Setting (identical to the sweep)
# ---------------------------------------------------------------------------
N            = 10
a0_in_sq     = 160.0
amplitude    = float(np.sqrt(a0_in_sq * N))
out_idx      = 2
KAPPA        = np.ones(N)
BETA         = 1e-3
T_TRANS, T_SS, DT = 40.0, 20.0, 0.05

SEEDS_A = list(range(8))
SEEDS_B = list(range(10))
GA0_LIST = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]


def det_ss(J, g, a_in, rng):
    """Simulate the (deterministic) dynamics and return steady-state"""
    if g == 0.0:
        return analytic_ss_g0(J, KAPPA, a_in)
    return simulate_record(J, g, KAPPA, a_in, None, T_TRANS, T_SS, DT,
                           0.0, a0=None, rng=rng).a_avg


def grad_matrices(J, g, seed=0):
    """Batched-over-XOR gradient estimates: SB sigma_y, SB sigma_x, vector field EP.

    All three share the free phase; sigma_x and EP also share the nudged phase
    (identical simulations — only the contraction differs by one conjugation).
    Symmetric-parameter bookkeeping (G + G^T - diag(G)) applied identically.
    """
    rng = np.random.default_rng(seed)
    sqk = np.sqrt(KAPPA)
    out = {k: np.zeros((N, N)) for k in ('sb_y', 'sb_x', 'ep')}

    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex)
        a_in[:2] = amplitude * inp
        a_ss = det_ss(J, g, a_in, rng)
        y = (a_in + sqk * a_ss)[out_idx].real
        e = y - tgt

        da_y = np.zeros(N, dtype=complex); da_y[out_idx] = -1j * BETA * e
        da_x = np.zeros(N, dtype=complex); da_x[out_idx] = BETA * e
        d_y = (det_ss(J, g, a_in + da_y, rng) - a_ss) / BETA
        d_x = (det_ss(J, g, a_in + da_x, rng) - a_ss) / BETA

        # unconstrained G_{jl}: response at j, source at l
        G_sb_y = -2.0 * np.real(np.outer(d_y, a_ss))          # Re[da_j abar_l]
        G_sb_x = -2.0 * np.imag(np.outer(d_x, a_ss))          # Im[da_j abar_l]
        G_ep   = +2.0 * np.imag(np.outer(d_x, a_ss.conj()))   # Im[da_j abar_l*]

        for key, G in (('sb_y', G_sb_y), ('sb_x', G_sb_x), ('ep', G_ep)):
            out[key] += G + G.T - np.diag(np.diag(G))         # J = J^T chain rule

    for key in out:
        out[key] /= len(XOR_INPUTS)
    return out


def ep_gradient_analytic_g0(J, seed=0):
    """Closed-form vector-field-EP estimate at g = 0: the IFT reference with the
    adjoint solve using Jac instead of Jac^T (EP's P = identity replaces the
    sigma_z conjugation, i.e. the transpose, in the resolvent). Validates the
    simulated EP implementation."""
    sqk = np.sqrt(KAPPA)
    grad = np.zeros((N, N))
    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex)
        a_in[:2] = amplitude * inp
        a_ss = analytic_ss_g0(J, KAPPA, a_in)
        u, v = a_ss.real, a_ss.imag
        y = (a_in + sqk * a_ss)[out_idx].real
        err = y - tgt
        Jac = np.block([[np.diag(-KAPPA / 2.0), J],
                        [-J, np.diag(-KAPPA / 2.0)]])
        e_L = np.zeros(2 * N)
        e_L[out_idx] = 2.0 * err * sqk[out_idx]
        lam = np.linalg.solve(Jac, e_L)          # <-- Jac, NOT Jac.T (EP)
        lam_u, lam_v = lam[:N], lam[N:]
        G = (-(np.outer(lam_u, v) + np.outer(v, lam_u))
             + (np.outer(lam_v, u) + np.outer(u, lam_v)))
        G[np.diag_indices_from(G)] *= 0.5
        grad += G
    return grad / len(XOR_INPUTS)


def main():
    # implementation check: simulated EP vs its own closed form (g = 0)
    J0 = _make_J(0)
    c_impl = _cosine(grad_matrices(J0, 0.0, seed=0)['ep'],
                     ep_gradient_analytic_g0(J0))
    print(f"EP implementation check (g=0): cos(sim, analytic) = {c_impl:+.6f}")
    assert c_impl > 0.999, "EP implementation does not match its closed form"

    # Panel (a): cosine vs ga0, standard J
    print(f"\nPanel (a) (N={N}, XOR, noiseless, {len(SEEDS_A)} seeds):")
    print(f"  {'ga0':>5} | {'SB sigma_y':>13} | {'SB sigma_x':>13} | {'EP':>13}")
    for ga0 in GA0_LIST:
        g = ga0 / a0_in_sq
        cs = {k: [] for k in cosA}
        for s in SEEDS_A:
            J = _make_J(s)
            ref = ift_gradient_batch(J, g, KAPPA, DT, seed=s)
            gm = grad_matrices(J, g, seed=s)
            for k in cs:
                cs[k].append(_cosine(gm[k], ref))
        for k in cosA:
            cosA[k].append((np.mean(cs[k]), np.std(cs[k]), np.array(cs[k])))
        print(f"  {ga0:5.1f} | " + " | ".join(
            f"{cosA[k][-1][0]:+.3f}±{cosA[k][-1][1]:.3f}" for k in cosA))

    # Panel (b): g = 0, cosine vs coupling scale (analytic, exact) 
    scales = np.linspace(0.05, 3.0, 30)
    cosB = {k: [] for k in ('sb_x', 'ep')}
    for s_val in scales:
        cs = {k: [] for k in cosB}
        for s in SEEDS_B:
            J = s_val * _make_J(s)
            ref = ift_gradient_batch(J, 0.0, KAPPA, DT, seed=s)
            gm = grad_matrices(J, 0.0, seed=s)
            for k in cosB:
                cs[k].append(_cosine(gm[k], ref))
        for k in cosB:
            cosB[k].append((np.mean(cs[k]), np.std(cs[k])))
    print(f"\nPanel (b) (g = 0, {len(SEEDS_B)} seeds): coupling scale "
          f"{scales[0]:.2f} -> {scales[-1]:.2f}: "
          f"EP {cosB['ep'][0][0]:+.3f} -> {cosB['ep'][-1][0]:+.3f}, "
          f"SB {cosB['sb_x'][0][0]:+.3f} -> {cosB['sb_x'][-1][0]:+.3f}")

    np.savez(os.path.join(HERE, 'Data', 'ep_vs_sb_data.npz'),
             ga0_list=GA0_LIST, seeds_a=SEEDS_A, scales=scales,
             seeds_b=SEEDS_B,
             cosA_sb_y=np.array([v[2] for v in cosA['sb_y']]),
             cosA_sb_x=np.array([v[2] for v in cosA['sb_x']]),
             cosA_ep=np.array([v[2] for v in cosA['ep']]),
             cosB_sb_x=np.array(cosB['sb_x']),
             cosB_ep=np.array(cosB['ep']))

    # Figure
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    labels = {'sb_y': r'SB, $U=\sigma_y$', 'sb_x': r'SB, $U=\sigma_x$',
              'ep': r'EP ($P=\mathbf{1}$)'}
    colors = {'sb_y': '#1f77b4', 'sb_x': '#2ca02c', 'ep': '#d62728'}

    ax = axes[0]
    for k in ('sb_y', 'sb_x', 'ep'):
        m = np.array([v[0] for v in cosA[k]])
        sd = np.array([v[1] for v in cosA[k]])
        ax.errorbar(GA0_LIST, m, yerr=sd, marker='o', ms=4, lw=1.4,
                    capsize=2.5, color=colors[k], label=labels[k])
        pts = np.array([v[2] for v in cosA[k]])            # (n_ga0, n_seeds)
        for i, ga0 in enumerate(GA0_LIST):
            ax.plot([ga0] * pts.shape[1], pts[i], '.', ms=2.5,
                    color=colors[k], alpha=0.35)
    ax.set_xlabel(r'$g\,\langle|a_\mathrm{in}|^2\rangle_\mathrm{avg}/\bar{\kappa}^2$')
    ax.set_ylabel(r'$\cos\!\left(\nabla^{\rm est}_\theta C,\ \nabla_\theta C\right)$')
    ax.axhline(0, color='k', lw=0.6, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.9)
    ax.set_title('(a) gradient fidelity vs nonlinearity', fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for k in ('sb_x', 'ep'):
        m = np.array([v[0] for v in cosB[k]])
        sd = np.array([v[1] for v in cosB[k]])
        ax.plot(scales, m, lw=1.6, color=colors[k], label=labels[k])
        ax.fill_between(scales, m - sd, m + sd, color=colors[k], alpha=0.18)
    ax.set_xlabel(r'coupling scale $s$   ($J = s\,J_0$,   $g = 0$)')
    ax.set_ylabel(r'$\cos\!\left(\nabla^{\rm est}_\theta C,\ \nabla_\theta C\right)$')
    ax.axhline(0, color='k', lw=0.6, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8.5, loc='lower right', framealpha=0.9)
    ax.set_title('(b) linear system, varying symplectic weight', fontsize=10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'fig_ep_vs_sb.{ext}'),
                    bbox_inches='tight', dpi=250)
    print(f"\nsaved {os.path.join(HERE, 'fig_ep_vs_sb.pdf')} (+ .png, + data npz)")


if __name__ == '__main__':
    main()
