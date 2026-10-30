"""
sc_core.py — noisy Kerr-network core, SEMICLASSICAL noise convention.

Conventions:

  1. Noise normalisation (symmetric ordering / truncated-Wigner):
         dW = (Z1 + i*Z2) * sqrt(dt/4),   Z ~ N(0,1)
         <dW dW^dagger> = dt/2 ,  <dW dW^T> = 0
     so a linear mode has vacuum occupation <|a|^2> = 1/2.

  2. Input-output relations with the noise included in the input field
     (Gardiner-Collett, per realisation):
         a_in(t)  = abar_in(x) + xi(t)          xi = dW/dt (white noise)
         a_out(t) = a_in(t) + sqrt(kappa) a(t)
     The measured record is a_out(t). The estimator subtracts only the known
     deterministic drive abar_in (the realisation of xi is not experimentally
     accessible):
         a_hat_Tm = (<a_out>_Tm - abar_in)/sqrt(kappa)
                  = <a>_Tm + (sum_k dW_k)/(sqrt(kappa) T_m).
     The SAME dW_k drives the dynamics and enters the record, so the
     interference between the promptly reflected vacuum and the
     cavity-re-emitted noise is automatic (output noise = S(omega) xi,
     |S| = 1 for a lossless network).

Physics (matching the paper / SI):
  da = [ -kappa/2 a - i J a - i g phi(a) - sqrt(kappa) abar_in ] dt - sqrt(kappa) dW
  phi_j(a) = (G @ |a|^2)_j a_j          (G = I: self-Kerr)
  J real symmetric, Delta_j := J_jj,  kappa_j > 0 for every mode (lossless
  otherwise: kappa' = 0).
  y   = Re[<a_out,out_idx>_Tm]          (network output, record-based)
  MSE loss, XOR encoding as in the paper: a_in,n = x_n |a0_in|, n = 1,2,
  |a0_in| = sqrt(N A), A = input power per node (paper: A = 160 kappa_bar).

SB perturbations (both kept):
  sigma_y : delta a_in[out] = -i beta err   -> dJ = -(2/beta) Re[sym outer]
  sigma_x : delta a_in[out] =  + beta err   -> dJ = -(2/beta) Im[sym outer]
Both are exact at g = 0 (linear system => linear response exact at any beta).

Integration: RK4 for the drift + Euler-Maruyama for the additive noise
(RK4-Maruyama). All simulation routines accept an optional batch dimension
(n_traj) for vectorised ensemble statistics.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def drift(a, J, g, kappa, a_in_det, G=None, nl=None, kappa_int=None):
    """Deterministic drift  -(kappa+kappa')/2 a - iJa + NL(a) - sqrt(kappa) abar_in.

    NL(a) defaults to the Kerr term -ig(G@|a|^2)a (G = I: self-Kerr). A custom
    nonlinearity can be supplied as a callable nl(a, g) returning the full
    complex NL drift contribution (e.g. saturable or two-photon absorption,
    cf. the 'other nonlinearities' study); G is then ignored.
    kappa_int (kappa', internal loss vector) defaults to 0; when nonzero, the
    FDT-matching vacuum enters through a SECOND independent noise channel in
    simulate_record. Works for a of shape (N,) or (n_traj, N).
    """
    if nl is not None:
        nl_term = nl(a, g)
    else:
        n2 = np.abs(a) ** 2
        kerr = n2 if G is None else n2 @ G.T
        nl_term = -1j * g * kerr * a
    damping = kappa if kappa_int is None else kappa + kappa_int
    return (-0.5 * damping * a
            - 1j * (a @ J.T)
            + nl_term
            - np.sqrt(kappa) * a_in_det)


def vacuum_noise(shape, dt, rng):
    """Wiener increment dW with <dW dW^dagger> = dt/2 per mode (semiclassical).

    Returns dW itself (NOT -sqrt(kappa) dW): the caller applies -sqrt(kappa) dW
    to the dynamics and accumulates the same dW in the output record.
    """
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)) * np.sqrt(dt / 4.0)


def rk4_step(a, dt, J, g, kappa, a_in_det, G=None, dW=None, nl=None,
             kappa_int=None, dW_int=None):
    """One RK4 drift step; EM noise -sqrt(kappa) dW (- sqrt(kappa') dW_int)."""
    k1 = drift(a,               J, g, kappa, a_in_det, G, nl, kappa_int)
    k2 = drift(a + 0.5 * dt * k1, J, g, kappa, a_in_det, G, nl, kappa_int)
    k3 = drift(a + 0.5 * dt * k2, J, g, kappa, a_in_det, G, nl, kappa_int)
    k4 = drift(a + dt * k3,       J, g, kappa, a_in_det, G, nl, kappa_int)
    a_new = a + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    if dW is not None:
        a_new = a_new - np.sqrt(kappa) * dW
    if dW_int is not None:
        a_new = a_new - np.sqrt(kappa_int) * dW_int
    return a_new


# ---------------------------------------------------------------------------
# Simulation of the measured record
# ---------------------------------------------------------------------------

class SimRecord:
    """Time averages over the sampling window [T_trans, T_trans + T_m].

    a_avg     : <a>_Tm                     (intracavity average)
    W_avg     : (sum_k dW_k)/T_m           (time-average of the input noise xi)
    n2_avg    : <|a_j|^2>_Tm               (per-mode second moment)
    a_out_avg : <a_out>_Tm = abar_in + W_avg + sqrt(kappa) a_avg   (record)
    a_hat     : (a_out_avg - abar_in)/sqrt(kappa)   (record-based estimator)
    a_end     : a(T_trans + T_m)           (for warm starts)
    """
    __slots__ = ('a_avg', 'W_avg', 'n2_avg', 'a_out_avg', 'a_hat', 'a_end')

    def __init__(self, a_avg, W_avg, n2_avg, a_out_avg, a_hat, a_end):
        self.a_avg, self.W_avg, self.n2_avg = a_avg, W_avg, n2_avg
        self.a_out_avg, self.a_hat, self.a_end = a_out_avg, a_hat, a_end


def simulate_record(J, g, kappa, a_in_det, G=None,
                    T_trans=40.0, T_m=40.0, dt=0.05,
                    noise_scale=1.0, a0=None, rng=None, n_traj=None, nl=None,
                    kappa_int=None):
    """Integrate the SDE and return the measured-record time averages.

    noise_scale = 0 -> deterministic (tests, gradient checks); the record then
    coincides with the intracavity average. n_traj batches independent
    trajectories (leading axis), all driven by the same deterministic a_in_det
    unless a_in_det itself carries a batch axis.

    kappa_int (kappa') adds internal loss with its own independent FDT vacuum
    channel (scaled by the same noise_scale); the internal-channel noise never
    enters the measured record (it leaves through the unmonitored loss port),
    so the record keeps only the collection-port dW. kappa_int=None reproduces
    the lossless behaviour exactly (no extra rng draws).
    """
    N = len(kappa)
    if rng is None:
        rng = np.random.default_rng()
    shape = (N,) if n_traj is None else (n_traj, N)
    if a0 is None:
        a0 = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    a = np.broadcast_to(np.asarray(a0, dtype=complex), shape).copy()
    sqk = np.sqrt(kappa)

    n_burn = max(1, int(round(T_trans / dt)))
    n_samp = max(1, int(round(T_m / dt)))

    for _ in range(n_burn):
        dW = noise_scale * vacuum_noise(shape, dt, rng)
        dW_i = (noise_scale * vacuum_noise(shape, dt, rng)
                if kappa_int is not None else None)
        a = rk4_step(a, dt, J, g, kappa, a_in_det, G, dW, nl, kappa_int, dW_i)

    S_a = np.zeros(shape, dtype=complex)
    S_W = np.zeros(shape, dtype=complex)
    S_n = np.zeros(shape, dtype=float)
    for _ in range(n_samp):
        dW = noise_scale * vacuum_noise(shape, dt, rng)
        dW_i = (noise_scale * vacuum_noise(shape, dt, rng)
                if kappa_int is not None else None)
        a = rk4_step(a, dt, J, g, kappa, a_in_det, G, dW, nl, kappa_int, dW_i)
        S_a += a * dt
        S_W += dW
        S_n += np.abs(a) ** 2 * dt

    T = n_samp * dt
    a_avg, W_avg, n2_avg = S_a / T, S_W / T, S_n / T
    a_out_avg = a_in_det + W_avg + sqk * a_avg
    a_hat = a_avg + W_avg / sqk
    return SimRecord(a_avg, W_avg, n2_avg, a_out_avg, a_hat, a)


# ---------------------------------------------------------------------------
# Linear (g = 0) reference results
# ---------------------------------------------------------------------------

def analytic_ss_g0(J, kappa, a_in_det):
    """Exact steady state at g = 0:  M a_ss = sqrt(kappa) a_in,  M = -kappa/2 - iJ."""
    M = -np.diag(kappa / 2.0) - 1j * J
    return np.linalg.solve(M, np.sqrt(kappa) * a_in_det)


def linear_smatrix_g0(J, kappa):
    """Zero-frequency scattering matrix at g = 0:
    a_out = S a_in,  S = I + sqrt(kappa) M^{-1} sqrt(kappa),  M = -kappa/2 - iJ.
    Single mode: S = 1 - kappa/(kappa/2 + i Delta); |S| = 1 (lossless)."""
    N = len(kappa)
    M = -np.diag(kappa / 2.0) - 1j * J
    sqk = np.diag(np.sqrt(kappa))
    return np.eye(N) + sqk @ np.linalg.solve(M, sqk)


# ---------------------------------------------------------------------------
# SB gradient (identical formula to noise_rk4.py; inputs are now record-based)
# ---------------------------------------------------------------------------

def sb_gradient_matrix(a_hat_free, delta_a_hat, beta, mode='sigma_y'):
    """Symmetric SB gradient matrix dJ (N x N);  J -= lr * dJ  descends.

    sigma_y: delta a_in = -i beta err ; dJ = -(2/beta) Re[sym outer]
    sigma_x: delta a_in = +  beta err ; dJ = -(2/beta) Im[sym outer]
    Diagonal halved (paper Eqs. (4)-(5): each (j,j) counted once).
    """
    M_cplx = np.outer(a_hat_free, delta_a_hat) + np.outer(delta_a_hat, a_hat_free)
    M = M_cplx.real if mode == 'sigma_y' else M_cplx.imag
    M[np.diag_indices_from(M)] *= 0.5
    return -(2.0 / beta) * M


def exact_gradient_g0(J, kappa, a_in_det, out_idx, target, eps=1e-6):
    """Central finite-difference gradient of L = (y - target)^2 at g = 0,
    y = Re[a_out[out_idx]], using the exact linear steady state. Reference for
    the unit tests (symmetric perturbations, matching the trainable J)."""
    N = len(kappa)
    sqk = np.sqrt(kappa)

    def loss(Jm):
        a_ss = analytic_ss_g0(Jm, kappa, a_in_det)
        y = (a_in_det + sqk * a_ss)[out_idx].real
        return (y - target) ** 2

    grad = np.zeros((N, N))
    for j in range(N):
        for l in range(j, N):
            Jp, Jm_ = J.copy(), J.copy()
            Jp[j, l] += eps
            Jm_[j, l] -= eps
            if l != j:
                Jp[l, j] += eps
                Jm_[l, j] -= eps
            gjl = (loss(Jp) - loss(Jm_)) / (2 * eps)
            grad[j, l] = gjl
            grad[l, j] = gjl
    return grad


# ---------------------------------------------------------------------------
# Training (XOR, record-based measurement)
# ---------------------------------------------------------------------------

XOR_INPUTS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
XOR_TARGETS = np.array([0, 1, 1, 0], dtype=float)


def train(N, J0, kappa, g, A,
          beta, out_idx=2, G=None,
          num_epochs=500, target_scale=1.0, learning_rate=1e-3,
          T_trans=40.0, T_m=40.0, dt=0.05,
          noise_scale=1.0, mode='sigma_y',
          verbose=True, seed=None):
    """Train the Kerr network on XOR with SB feedback, record-based estimators.

    NOTE: this function applies ONLINE (per-pattern) updates and is NOT used for
    any figure in the paper. The published Sec. IX sweeps use train_fixed_ratio
    (batched update, one per epoch) via the gradient_worker_sc*.py scripts.

    A : input power per node, |a0_in| = sqrt(N A)   (paper: A = 160 kappa_bar)
    Returns loss_history, phot_coherent (|a_hat_j|^2 averaged over active
    inputs), phot_total (<|a_j|^2>_Tm, contains the +1/2 vacuum), ratio_history.
    """
    rng = np.random.default_rng(seed)
    J = J0.copy()
    amplitude = np.sqrt(A * N)

    loss_history, phot_c_history, phot_t_history, ratio_history = [], [], [], []

    for epoch in range(num_epochs):
        e_loss, e_phc, e_pht, e_ratio = 0.0, [], [], []

        for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
            a_in = np.zeros(N, dtype=complex)
            a_in[:2] = amplitude * inp

            rec = simulate_record(J, g, kappa, a_in, G, T_trans, T_m, dt,
                                  noise_scale, a0=None, rng=rng)
            y = rec.a_out_avg[out_idx].real
            err = y - tgt / target_scale
            e_loss += err ** 2 * target_scale ** 2

            if np.any(inp > 0.1):
                e_phc.append(np.mean(np.abs(rec.a_hat) ** 2))
                e_pht.append(np.mean(rec.n2_avg))

            delta_a_in = np.zeros(N, dtype=complex)
            delta_a_in[out_idx] = (-1j if mode == 'sigma_y' else 1.0) * beta * err

            if np.any(inp > 0.1):
                active_amp = amplitude * float(np.mean(inp[inp > 0.1]))
                e_ratio.append(abs(delta_a_in[out_idx]) / active_amp)

            rec_p = simulate_record(J, g, kappa, a_in + delta_a_in, G,
                                    T_trans, T_m, dt, noise_scale,
                                    a0=rec.a_end.copy(), rng=rng)

            dJ = sb_gradient_matrix(rec.a_hat, rec_p.a_hat - rec.a_hat,
                                    beta, mode=mode)
            J -= learning_rate * dJ

        e_loss /= len(XOR_INPUTS)
        loss_history.append(e_loss)
        phot_c_history.append(np.mean(e_phc) if e_phc else np.nan)
        phot_t_history.append(np.mean(e_pht) if e_pht else np.nan)
        ratio_history.append(float(np.mean(e_ratio)) if e_ratio else np.nan)

        if verbose and (epoch % max(1, num_epochs // 20) == 0 or epoch == num_epochs - 1):
            print(f"  epoch {epoch:4d}  loss={e_loss:.3e}  "
                  f"|a_hat|^2={phot_c_history[-1]:.1f}")

    return (np.array(loss_history), phot_c_history, phot_t_history, ratio_history)
