"""
drift_worker_sc.py — drift-RATE sweep during online training, SEMICLASSICAL
convention (sc_core.py physics).

Online SB training continues while the ring detunings random-walk each epoch (common-mode by
default); one job = one (arm, sigma_step, seed). All noisy simulations use the
record-based measurement (<dW dW^dagger> = dt/2, reflected input noise in the
record); deterministic evaluations are unchanged.

Arms:
  noisy  : start from J_noisy, noise_scale=1 (online)   <- main
  clean  : start from J_clean, noise_scale=0 (online)   <- main
  frozen : start from J_noisy, lr=0 (no learning)       <- reference

The pre-trained networks are NOT embedded (the old base64 networks were trained
under the previous convention and do not apply): they are loaded from the npz
produced by robustness_seed_worker_sc.py. Run that first (at least seed 0):

  python robustness_seed_worker_sc.py --seed 0
  python drift_worker_sc.py --arm noisy --sigma_step 0.03 --seed 0 \
      --npz results_robustness_seeds_sc/robustness_seed0.npz

Requires sc_core.py in the parent directory (or alongside on the cluster).
"""

import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (HERE, os.path.dirname(HERE),
           os.path.join(os.path.dirname(HERE), 'Noise_analysis'),
           os.path.join(os.path.dirname(os.path.dirname(HERE)), 'Noise_analysis')):  # shared modules (sc_core, gradient_worker_sc) live in Code/Noise_analysis
    if p_ not in sys.path:
        sys.path.insert(0, p_)
from sc_core import (simulate_record, sb_gradient_matrix,
                     XOR_INPUTS, XOR_TARGETS)
from robustness_seed_worker_sc import (DEFAULTS, EPS_ERR, _measure,
                                       evaluate_xor_loss)


def train_one_epoch(J, g, kappa, amplitude, out_idx, target_scale,
                    ratio, mode, lr, T_trans, T_sample, dt, noise_scale, rng):
    N = len(kappa)
    J_new = J.copy()
    epoch_loss = 0.0
    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        a_in = np.zeros(N, dtype=complex)
        a_in[:2] = amplitude * inp
        a0 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        rec = _measure(J_new, g, kappa, a_in, T_trans, T_sample, dt,
                       noise_scale, rng, a0=a0)
        y = rec.a_out_avg[out_idx].real
        err = y - tgt / target_scale
        epoch_loss += err ** 2 * target_scale ** 2
        beta_eff = ratio * amplitude / max(abs(err), EPS_ERR)
        da = np.zeros(N, dtype=complex)
        if mode == 'sigma_y':
            da[out_idx] = -1j * beta_eff * err
        else:
            da[out_idx] = beta_eff * err
        rec_p = _measure(J_new, g, kappa, a_in + da, T_trans, T_sample, dt,
                         noise_scale, rng, a0=rec.a_end.copy())
        J_new -= lr * sb_gradient_matrix(rec.a_hat, rec_p.a_hat - rec.a_hat,
                                         beta_eff, mode=mode)
    return J_new, epoch_loss / len(XOR_INPUTS)


def train_with_drift(J_start, g, kappa, p, noise_scale, sigma_step,
                     drift_mode='common', lr=None, seed=0,
                     eval_noise_scale=0.0):
    """Online SB training while the detunings random-walk each epoch."""
    rng   = np.random.default_rng(seed)
    rng_d = np.random.default_rng(seed + 20000)
    J     = J_start.copy()
    N     = p['N']
    amp   = np.sqrt(p['a0_in_squared'] * N)
    lr    = p['lr'] if lr is None else lr
    T_train = p['T_sample_noisy'] if noise_scale > 0 else p['T_sample_clean']

    train_lh, eval_lh, drift_trace = [], [], []
    accum = 0.0
    for _epoch in range(p['num_epochs']):
        if sigma_step > 0:
            if drift_mode == 'common':
                s = float(sigma_step * rng_d.standard_normal())
                J[np.diag_indices(N)] += s
                accum += s
            else:
                s_vec = sigma_step * rng_d.standard_normal(N)
                J[np.diag_indices(N)] += s_vec
                accum = float(np.sqrt(accum ** 2 + np.mean(s_vec ** 2)))
        drift_trace.append(accum)

        if lr > 0:
            J, epoch_loss = train_one_epoch(
                J, g, kappa, amp, p['out_idx'], p['target_scale'],
                p['ratio'], p['mode'], lr,
                p['T_trans'], T_train, p['dt'], noise_scale, rng)
        else:
            epoch_loss = evaluate_xor_loss(
                J, g, kappa, amp, out_idx=p['out_idx'],
                target_scale=p['target_scale'], T_trans=p['T_trans'],
                T_sample=T_train, dt=p['dt_eval'],
                noise_scale=noise_scale, n_seeds=1, rng=rng)
        train_lh.append(epoch_loss)

        eval_loss = evaluate_xor_loss(
            J, g, kappa, amp, out_idx=p['out_idx'],
            target_scale=p['target_scale'], T_trans=p['T_trans'],
            T_sample=p['T_sample_clean'], dt=p['dt_eval'],
            noise_scale=eval_noise_scale, n_seeds=1)
        eval_lh.append(eval_loss)

    return np.array(train_lh), np.array(eval_lh), np.array(drift_trace), J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['noisy', 'clean', 'frozen'],
                    required=True)
    ap.add_argument('--sigma_step', type=float, required=True,
                    help='Per-epoch RMS detuning drift [kappa/epoch].')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--num_epochs',     type=int,   default=200)
    ap.add_argument('--drift_mode',     default='common',
                    choices=['common', 'independent'])
    ap.add_argument('--transient_frac', type=float, default=0.4)
    ap.add_argument('--npz', default=os.path.join(
        HERE, 'results_robustness_seeds_sc', 'robustness_seed0.npz'),
        help='Pre-trained networks (J_clean/J_noisy/g/kappa) from '
             'robustness_seed_worker_sc.py.')
    ap.add_argument('--out_dir', default=os.path.join(
        HERE, 'results_drift_sweep_sc'))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    p = dict(DEFAULTS)
    p['num_epochs'] = args.num_epochs

    if not os.path.exists(args.npz):
        raise SystemExit(f"ERROR: --npz {args.npz} not found. Run "
                         "robustness_seed_worker_sc.py first (seed 0).")
    d = np.load(args.npz)
    J_clean = d['J_clean']; J_noisy = d['J_noisy']
    g = float(d['g']); kappa = d['kappa']

    if args.arm == 'noisy':
        J0, noise_scale, lr = J_noisy, 1.0, p['lr']
    elif args.arm == 'clean':
        J0, noise_scale, lr = J_clean, 0.0, p['lr']
    else:  # frozen
        J0, noise_scale, lr = J_noisy, 1.0, 0.0

    train_lh, eval_lh, drift_trace, _ = train_with_drift(
        J0, g, kappa, p, noise_scale=noise_scale, sigma_step=args.sigma_step,
        drift_mode=args.drift_mode, lr=lr, seed=args.seed)

    n0 = int(args.transient_frac * args.num_epochs)
    ss = eval_lh[n0:]
    stats = dict(
        convention='semiclassical_record',
        arm=args.arm, sigma_step=float(args.sigma_step), seed=int(args.seed),
        num_epochs=int(args.num_epochs),
        t_sample_clean=float(p['T_sample_clean']),
        t_sample_noisy=float(p['T_sample_noisy']),
        ratio=float(p['ratio']), lr=float(p['lr']), dt=float(p['dt']),
        drift_mode=args.drift_mode,
        sigma_total=float(args.sigma_step * np.sqrt(args.num_epochs)),
        mse_mean=float(np.mean(ss)), mse_median=float(np.median(ss)),
        mse_p90=float(np.percentile(ss, 90)), mse_max=float(np.max(ss)),
        frac_above_1e2=float(np.mean(ss > 1e-2)),
        frac_above_1e1=float(np.mean(ss > 1e-1)),
        drift_rms_tail=float(np.sqrt(np.mean(drift_trace[n0:] ** 2))),
        drift_abs_final=float(abs(drift_trace[-1])),
    )

    fname = f"{args.arm}_sstep{args.sigma_step:.5f}_seed{args.seed}.npz"
    np.savez(os.path.join(args.out_dir, fname),
             eval_lh=eval_lh, train_lh=train_lh, drift_trace=drift_trace,
             **stats)
    print(f"[{args.arm}] sstep={args.sigma_step:.4f} seed={args.seed}  "
          f"mse_mean={stats['mse_mean']:.3e}  p90={stats['mse_p90']:.3e}  "
          f"frac>1e-2={stats['frac_above_1e2']:.2f}  "
          f"frac>1e-1={stats['frac_above_1e1']:.2f}", flush=True)


if __name__ == '__main__':
    main()
