"""
robustness_seed_worker_sc.py — one cluster job = the FULL static robustness
study for ONE J0 seed, SEMICLASSICAL convention (sc_core.py physics).

Conventions:
  - noise <dW dW^dagger> = dt/2 (truncated Wigner; vacuum <|a|^2> = 1/2);
  - record-based estimators: the measured output is
    <a_out>_Tm = abar_in + (Sum dW_k)/T_m + sqrt(kappa) <a>_Tm (reflected input
    noise included), and only the calibrated deterministic drive is subtracted;
  - thermal bath: noise_scale = sqrt(1 + 2 n_th) — a noise-POWER ratio, hence
    unchanged by the convention switch; under symmetric ordering it is exactly
    the FDT correlator <dW_th dW_th^dagger> = (n_th + 1/2) dt, and the g = 0
    stationary occupation is n_th + 1/2 (verified in ../test_sc_core.py, T10).


Everything else matches the noise 'sweep worker': N = 10, a0_in_sq = 160
(amplitude 40), fixed nudge ratio 0.04, ONLINE sigma_y SB, lr = 5e-4,
800 epochs, dt = 0.05 (train) / 0.025 (eval), T = 20 clean / 200 noisy,
locked operating point g = 1/320 (g<|a_in|^2>/kappa^2 = 1/2), identical
disorder grids and perturbation seed bases. Panels: detuning, coupling,
loss-rate, nonlinearity disorder, cross-talk, thermal, readout phase.

Requires sc_core.py in the parent directory (or alongside on the cluster).

Thermal-noise row of the SI table: the high-drive rerun that produced
results_thermal_noise_sc/thermal_a0_1600_5seed_data.npz is reproduced with
  --a0_in_sq 1600 --g 3.125e-4
(defaults --a0_in_sq 160, g = 1/320 = 0.003125).

Usage
-----
  python robustness_seed_worker_sc.py --seed 0 --n_pert 15 --n_sigma 15 \
      --num_epochs 800 --out_dir results_robustness_seeds_sc
"""

import os
import sys 
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (HERE, os.path.dirname(HERE),
           os.path.join(os.path.dirname(HERE), 'Noise_analysis')):  # shared modules (sc_core) live in Code/Noise_analysis
    if p_ not in sys.path:
        sys.path.insert(0, p_)
from sc_core import (simulate_record, analytic_ss_g0, sb_gradient_matrix,
                     XOR_INPUTS, XOR_TARGETS)

EPS_ERR = 1e-6

DEFAULTS = dict(
    N=10, chi_ss=1.0, a0_in_squared=160.0, kappa_val=1.0,
    ratio=0.04, mode='sigma_y', out_idx=2, target_scale=1.0,
    num_epochs=800, lr=5e-4,
    T_trans=40.0, T_sample_clean=20.0, T_sample_noisy=200.0, dt=0.05,
    # Finer eval step: strongly-perturbed J (sigma up to 10 kappa) makes
    # explicit RK4 unstable at dt=0.05; dt=0.025 is stable (cf. sweep-21 note).
    dt_eval=0.025,
)

# ---------------------------------------------------------------------------
# Measurement helpers (record-based)
# ---------------------------------------------------------------------------

def _measure(J, g, kappa, a_in, T_trans, T_sample, dt, noise_scale, rng,
             a0=None, nl=None):
    return simulate_record(J, g, kappa, a_in, None, T_trans, T_sample, dt,
                           noise_scale, a0=a0, rng=rng, nl=nl)


def _y_measured(J, g, kappa, a_in, out_idx, T_trans, T_sample, dt,
                noise_scale, rng, nl=None):
    """Record-based network output for one pattern."""
    if g == 0.0 and noise_scale == 0.0 and nl is None:
        a_ss = analytic_ss_g0(J, kappa, a_in)
        return (a_in + np.sqrt(kappa) * a_ss)[out_idx].real
    rec = _measure(J, g, kappa, a_in, T_trans, T_sample, dt, noise_scale, rng,
                   nl=nl)
    return rec.a_out_avg[out_idx].real


def evaluate_xor_loss(J, g, kappa, amplitude, out_idx=2, target_scale=1.0,
                      T_trans=20.0, T_sample=100.0, dt=0.025,
                      noise_scale=0.0, n_seeds=1, rng=None, nl=None):
    if rng is None:
        rng = np.random.default_rng()
    N = len(kappa)
    total = 0.0
    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex)
        a_in[:2] = amplitude * inp
        ys = [_y_measured(J, g, kappa, a_in, out_idx, T_trans, T_sample, dt,
                          noise_scale, rng, nl=nl) for _ in range(n_seeds)]
        err = float(np.mean(ys)) - tgt / target_scale
        total += err ** 2 * target_scale ** 2
    return total / len(XOR_INPUTS)

# ---------------------------------------------------------------------------
# Perturbations (identical to the noise sweep worker) 
# ---------------------------------------------------------------------------
def perturb_detunings(J, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    J_new = J.copy(); N = J.shape[0]
    J_new[np.diag_indices(N)] += sigma * rng.standard_normal(N)
    return J_new


def perturb_couplings(J, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    N = J.shape[0]
    raw = sigma * rng.standard_normal((N, N))
    noise = (raw + raw.T) / np.sqrt(2)
    np.fill_diagonal(noise, 0.0)
    return J + noise


def perturb_kappa(kappa, sigma_rel, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    scale = 1.0 + sigma_rel * rng.standard_normal(len(kappa))
    return kappa * np.maximum(scale, 0.01)


def perturb_g(g, sigma_rel, N, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return np.maximum(g * (1.0 + sigma_rel * rng.standard_normal(N)), 0.0)


def perturb_crosstalk(J, sigma, n_pairs=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    N = J.shape[0]
    if n_pairs is None:
        n_pairs = max(1, N // 2)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    chosen = rng.choice(len(all_pairs), size=min(n_pairs, len(all_pairs)),
                        replace=False)
    J_new = J.copy()
    for idx in chosen:
        i, j = all_pairs[int(idx)]
        c = sigma * float(rng.standard_normal())
        J_new[i, j] += c; J_new[j, i] += c
    return J_new


def make_g_vector_nl(g_vec):
    """Per-mode Kerr disorder via sc_core's pluggable nonlinearity."""
    def nl_fn(a, g_):
        return -1j * g_vec * np.abs(a) ** 2 * a
    return nl_fn


def n_th_to_noise_scale(n_th):
    """noise-POWER ratio (n_th + 1/2)/(1/2) -> amplitude sqrt(1 + 2 n_th).
    Convention-independent; = symmetric-ordering FDT under sc_core."""
    return float(np.sqrt(1.0 + 2.0 * max(n_th, 0.0)))


def compute_g_from_chiss(J, chi_ss, kappa, amplitude, out_idx):
    N = len(kappa)
    phots = []
    for inp in XOR_INPUTS:
        if not np.any(inp > 0.1):
            continue
        a_in = np.zeros(N, dtype=complex)
        a_in[:2] = amplitude * inp
        a_ss = analytic_ss_g0(J, kappa, a_in)
        phots.append(np.mean(np.abs(a_ss) ** 2))
    mean_phot_g0 = float(np.mean(phots)) if phots else 1.0
    return chi_ss / max(mean_phot_g0, 1e-12)

# ---------------------------------------------------------------------------
# Training (online, fixed-ratio-based, record-based measurement)
# ---------------------------------------------------------------------------
def train_system(J0, g, kappa, p, noise_scale, seed):
    rng = np.random.default_rng(seed)
    J = J0.copy(); N = p['N']; amp = np.sqrt(p['a0_in_squared'] * N); lh = []
    T_s = p['T_sample_noisy'] if noise_scale > 0 else p['T_sample_clean']
    for _epoch in range(p['num_epochs']):
        el = 0.0
        for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
            a_in = np.zeros(N, dtype=complex)
            a_in[:2] = amp * inp
            a0 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            rec = _measure(J, g, kappa, a_in, p['T_trans'], T_s, p['dt'],
                           noise_scale, rng, a0=a0)
            y = rec.a_out_avg[p['out_idx']].real
            err = y - tgt / p['target_scale']
            el += err ** 2 * p['target_scale'] ** 2
            beta_eff = p['ratio'] * amp / max(abs(err), EPS_ERR)
            da = np.zeros(N, dtype=complex)
            if p['mode'] == 'sigma_y':
                da[p['out_idx']] = -1j * beta_eff * err
            else:
                da[p['out_idx']] = beta_eff * err
            rec_p = _measure(J, g, kappa, a_in + da, p['T_trans'], T_s,
                             p['dt'], noise_scale, rng, a0=rec.a_end.copy())
            J -= p['lr'] * sb_gradient_matrix(rec.a_hat,
                                              rec_p.a_hat - rec.a_hat,
                                              beta_eff, mode=p['mode'])
        lh.append(el / len(XOR_INPUTS))
    return J, np.array(lh)

# ---------------------------------------------------------------------------
# Disorder sweeps (protocol and seed bases identical to noise sweep)
# ---------------------------------------------------------------------------
def _eval_loss_standard(J, g, kappa, amp, p, rng):
    return evaluate_xor_loss(J, g, kappa, amp, out_idx=p['out_idx'],
                             target_scale=p['target_scale'],
                             T_trans=p['T_trans'],
                             T_sample=p['T_sample_clean'], dt=p['dt_eval'],
                             noise_scale=0.0, n_seeds=1, rng=rng)


def sweep_J_perturbation(J, sigma_values, perturb_fn, g, kappa, amp, p, n_pert,
                         seed_base=9000, **fn_kwargs):
    means, stds = [], []
    for sigma in sigma_values:
        losses = []
        for k in range(n_pert):
            rng = np.random.default_rng(seed_base + k)
            J_p = perturb_fn(J, sigma, rng=rng, **fn_kwargs)
            losses.append(_eval_loss_standard(J_p, g, kappa, amp, p, rng))
        means.append(np.mean(losses)); stds.append(np.std(losses))
    return np.array(means), np.array(stds)


def sweep_kappa(J, sigma_values, g, kappa_base, amp, p, n_pert, seed_base=8000):
    means, stds = [], []
    for sigma in sigma_values:
        losses = []
        for k in range(n_pert):
            rng = np.random.default_rng(seed_base + k)
            kp = perturb_kappa(kappa_base, sigma, rng=rng)
            losses.append(_eval_loss_standard(J, g, kp, amp, p, rng))
        means.append(np.mean(losses)); stds.append(np.std(losses))
    return np.array(means), np.array(stds)


def sweep_g_disorder(J, sigma_values, g, kappa, amp, p, n_pert, seed_base=7000):
    N = p['N']; means, stds = [], []
    for sigma in sigma_values:
        losses = []
        for k in range(n_pert):
            rng = np.random.default_rng(seed_base + k)
            g_vec = perturb_g(g, sigma, N, rng=rng)
            losses.append(evaluate_xor_loss(
                J, g, kappa, amp,   # scalar g unused here: nl carries g_vec
                out_idx=p['out_idx'],
                target_scale=p['target_scale'], T_trans=p['T_trans'],
                T_sample=p['T_sample_clean'], dt=p['dt_eval'],
                noise_scale=0.0, n_seeds=1, rng=rng,
                nl=make_g_vector_nl(g_vec)))
        means.append(np.mean(losses)); stds.append(np.std(losses))
    return np.array(means), np.array(stds)


def sweep_thermal(J, n_th_values, g, kappa, amp, p, n_seeds_per_point,
                  seed_base=6000):
    means, stds = [], []
    for n_th in n_th_values:
        ns = n_th_to_noise_scale(n_th)
        losses = []
        for k in range(n_seeds_per_point):
            rng = np.random.default_rng(seed_base + k)
            losses.append(evaluate_xor_loss(
                J, g, kappa, amp, out_idx=p['out_idx'],
                target_scale=p['target_scale'], T_trans=p['T_trans'],
                T_sample=p['T_sample_noisy'], dt=p['dt_eval'],
                noise_scale=ns, n_seeds=3, rng=rng))
        means.append(np.mean(losses)); stds.append(np.std(losses))
    return np.array(means), np.array(stds)


def sweep_phase(J, phi_values, g, kappa, amp, p, n_pert, seed_base=5000):
    """Inference-only readout phase noise: the measured field is
    a_ss e^{i phi}, phi ~ N(0, sigma_phi^2), independent per pattern.
    Deterministic steady state (noise 0: record = intracavity) computed once
    per pattern, phase applied post-hoc."""
    N = p['N']; out = p['out_idx']; ts = p['target_scale']
    a_ss_pat = []
    for inp in XOR_INPUTS:
        a_in = np.zeros(N, dtype=complex); a_in[:2] = amp * inp
        rec = _measure(J, g, kappa, a_in, p['T_trans'], p['T_sample_clean'],
                       p['dt_eval'], 0.0, np.random.default_rng(0))
        a_ss_pat.append((a_in, rec.a_avg))
    means, stds = [], []
    for sphi in phi_values:
        losses = []
        for k in range(n_pert):
            rng = np.random.default_rng(seed_base + k)
            tot = 0.0
            for (a_in, a_ss), tgt in zip(a_ss_pat, XOR_TARGETS):
                phi = sphi * rng.standard_normal()
                y = (a_in + np.sqrt(kappa) * a_ss * np.exp(1j * phi))[out].real
                tot += (y - tgt / ts) ** 2 * ts ** 2
            losses.append(tot / len(XOR_INPUTS))
        means.append(np.mean(losses)); stds.append(np.std(losses))
    return np.array(means), np.array(stds)

# ---------------------------------------------------------------------------
# Main Worker
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed',       type=int,   required=True)
    ap.add_argument('--n_pert',     type=int,   default=15)
    ap.add_argument('--n_sigma',    type=int,   default=15)
    ap.add_argument('--num_epochs', type=int,   default=DEFAULTS['num_epochs'])
    ap.add_argument('--ratio',      type=float, default=DEFAULTS['ratio'])
    ap.add_argument('--t_sample_clean', type=float,
                    default=DEFAULTS['T_sample_clean'])
    ap.add_argument('--t_sample_noisy', type=float,
                    default=DEFAULTS['T_sample_noisy'])
    # Drive power per node and nonlinearity. Defaults reproduce the main study
    # (a0_in_sq = 160, g = 0.5/160 = 1/320 = 0.003125). The SI thermal-noise row
    # was produced with:  --a0_in_sq 1600 --g 3.125e-4
    # (--g omitted keeps the locked operating point g = 0.5*kappa/a0_in_sq,
    # which gives the same values at both drive powers).
    ap.add_argument('--a0_in_sq', type=float,
                    default=DEFAULTS['a0_in_squared'])
    ap.add_argument('--g', type=float, default=None)
    ap.add_argument('--out_dir', default=os.path.join(
        HERE, 'results_robustness_seeds_sc'))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    p = dict(DEFAULTS)
    p['num_epochs']     = args.num_epochs
    p['ratio']          = args.ratio
    p['T_sample_clean'] = args.t_sample_clean
    p['T_sample_noisy'] = args.t_sample_noisy
    p['a0_in_squared']  = args.a0_in_sq
    N     = p['N']
    kappa = np.ones(N) * p['kappa_val']
    amp   = np.sqrt(p['a0_in_squared'] * N)

    rng_init = np.random.default_rng(args.seed)
    R  = rng_init.standard_normal((N, N))
    J0 = (R + R.T) / np.sqrt(2 * N)
    # Locked operating point: g<|a_in|^2>/kappa^2 = 1/2 => g = 1/320 at the
    # default a0_in_sq = 160 (overridable with --g).
    g = args.g if args.g is not None else 0.5 * p['kappa_val'] / p['a0_in_squared']
    chi_eff = g / compute_g_from_chiss(J0, 1.0, kappa, amp, p['out_idx'])
    print(f"seed={args.seed}  a0_in_sq={p['a0_in_squared']:g}  g={g:.6g}  "
          f"chi_ss(intracav)={chi_eff:.2f}", flush=True)

    J_clean, lh_clean = train_system(J0, g, kappa, p, noise_scale=0.0,
                                     seed=args.seed)
    print(f"  trained clean:  final loss={lh_clean[-1]:.3e}", flush=True)
    J_noisy, lh_noisy = train_system(J0, g, kappa, p, noise_scale=1.0,
                                     seed=args.seed)
    print(f"  trained noisy:  final loss={lh_noisy[-1]:.3e}", flush=True)

    loss0_c = _eval_loss_standard(J_clean, g, kappa, amp, p,
                                  np.random.default_rng(1))
    loss0_n = _eval_loss_standard(J_noisy, g, kappa, amp, p,
                                  np.random.default_rng(1))

    sig_J   = np.logspace(-3, 1, args.n_sigma)
    sig_rel = np.logspace(-3, 0, args.n_sigma)
    n_th_v  = np.logspace(-2, 1, args.n_sigma)
    phi_v   = np.logspace(-3, np.log10(np.pi / 2), args.n_sigma)
    n_pairs = max(1, N // 2)

    out = dict(convention='semiclassical_record',
               seed=args.seed, g=g, kappa=kappa, J0=J0,
               J_clean=J_clean, J_noisy=J_noisy,
               lh_clean=lh_clean, lh_noisy=lh_noisy,
               loss0_clean=loss0_c, loss0_noisy=loss0_n,
               sig_J=sig_J, sig_rel=sig_rel, n_th_v=n_th_v, phi_v=phi_v)

    def do(tag, fn_c, fn_n):
        print(f"  sweep {tag} ...", flush=True)
        mc, sc = fn_c(); mn, sn = fn_n()
        out[f'{tag}_m_c'] = mc; out[f'{tag}_s_c'] = sc
        out[f'{tag}_m_n'] = mn; out[f'{tag}_s_n'] = sn

    do('detuning',
       lambda: sweep_J_perturbation(J_clean, sig_J, perturb_detunings, g, kappa, amp, p, args.n_pert, seed_base=9000),
       lambda: sweep_J_perturbation(J_noisy, sig_J, perturb_detunings, g, kappa, amp, p, args.n_pert, seed_base=9000))
    do('coupling',
       lambda: sweep_J_perturbation(J_clean, sig_J, perturb_couplings, g, kappa, amp, p, args.n_pert, seed_base=8100),
       lambda: sweep_J_perturbation(J_noisy, sig_J, perturb_couplings, g, kappa, amp, p, args.n_pert, seed_base=8100))
    do('kappa',
       lambda: sweep_kappa(J_clean, sig_rel, g, kappa, amp, p, args.n_pert),
       lambda: sweep_kappa(J_noisy, sig_rel, g, kappa, amp, p, args.n_pert))
    do('g_disorder',
       lambda: sweep_g_disorder(J_clean, sig_rel, g, kappa, amp, p, args.n_pert),
       lambda: sweep_g_disorder(J_noisy, sig_rel, g, kappa, amp, p, args.n_pert))
    do('crosstalk',
       lambda: sweep_J_perturbation(J_clean, sig_J, perturb_crosstalk, g, kappa, amp, p, args.n_pert, seed_base=7200, n_pairs=n_pairs),
       lambda: sweep_J_perturbation(J_noisy, sig_J, perturb_crosstalk, g, kappa, amp, p, args.n_pert, seed_base=7200, n_pairs=n_pairs))
    do('thermal',
       lambda: sweep_thermal(J_clean, n_th_v, g, kappa, amp, p, n_seeds_per_point=args.n_pert),
       lambda: sweep_thermal(J_noisy, n_th_v, g, kappa, amp, p, n_seeds_per_point=args.n_pert))
    do('phase',
       lambda: sweep_phase(J_clean, phi_v, g, kappa, amp, p, args.n_pert),
       lambda: sweep_phase(J_noisy, phi_v, g, kappa, amp, p, args.n_pert))

    fname = os.path.join(args.out_dir, f'robustness_seed{args.seed}.npz')
    np.savez(fname, **out)
    print(f"Saved  {fname}", flush=True)


if __name__ == '__main__':
    main()
