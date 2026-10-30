"""
phase_train_worker_sc.py — train under readout phase noise, SEMICLASSICAL
convention (sc_core.py physics, record-based estimators).

Port of Corrected_XOR/robustness/phase_train_worker.py: one network is trained
from J0(seed) with full-homodyne readout phase noise present DURING training —
every steady-state measurement acquires a common phase error phi ~ N(0,
sigma_phi^2), independent per measurement, corrupting both the error signal
AND the SB gradient. Online (or batch) sigma_y updates, fixed nudge ratio
0.04, g = 1/320.

Phase-error model (choice made explicit under the record convention):
  default          : the phase rotates the field ESTIMATOR, a_hat -> a_hat e^{i phi}
                     (protocol-faithful port of the old a_ss e^{i phi}; the
                     drive subtraction is assumed phase-clean).
  --rotate_record  : the LO phase rotates the ENTIRE record,
                     a_hat_meas = (e^{i phi} <a_out>_Tm - abar_in)/sqrt(kappa),
                     which additionally leaks the reflected drive into the
                     measurement at the driven nodes (|abar_in| phi / sqrt(k))
                     — the physically stricter model. y at the (undriven)
                     output node is identical in both models.

Saves the (phase-corrupted) training loss history and the final CLEAN
phase-free eval loss of the trained network.

Usage
-----
  python phase_train_worker_sc.py --seed 0 --sigma_phi 0.1 --noise 1 \
      --num_epochs 800 --out_dir results_phase_train_sc
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
from sc_core import sb_gradient_matrix, XOR_INPUTS, XOR_TARGETS
from robustness_seed_worker_sc import (DEFAULTS, EPS_ERR, _measure,
                                       evaluate_xor_loss)


def measured_fields(rec, a_in, kappa, phi, rotate_record):
    """Apply the readout phase error; return (a_hat_meas, y_full_record)."""
    if rotate_record:
        rot_out = rec.a_out_avg * np.exp(1j * phi)
        a_hat_m = (rot_out - a_in) / np.sqrt(kappa)
    else:
        a_hat_m = rec.a_hat * np.exp(1j * phi)
    return a_hat_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--sigma_phi', type=float, required=True,
                    help='readout phase noise [rad]')
    ap.add_argument('--noise', type=int, default=1, choices=[0, 1])
    ap.add_argument('--batch', type=int, default=0, choices=[0, 1],
                    help='0 online / 1 batch (mean gradient per epoch)')
    ap.add_argument('--rotate_record', action='store_true',
                    help='rotate the full record (incl. reflected drive) '
                         'instead of the field estimator only')
    ap.add_argument('--num_epochs', type=int, default=800)
    ap.add_argument('--g', type=float, default=0.003125)
    ap.add_argument('--out_dir', default=os.path.join(
        HERE, 'results_phase_train_sc'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    p = dict(DEFAULTS); N = p['N']; out = p['out_idx']; ts = p['target_scale']
    kappa = np.ones(N) * p['kappa_val']; amp = np.sqrt(p['a0_in_squared'] * N)
    g = args.g; Tt = p['T_trans']; ratio = p['ratio']; dt = p['dt']
    lr = p['lr']
    noise_scale = float(args.noise)
    Ts = p['T_sample_noisy'] if noise_scale > 0 else p['T_sample_clean']
    sphi = args.sigma_phi

    rng0 = np.random.default_rng(args.seed)
    R = rng0.standard_normal((N, N)); J = (R + R.T) / np.sqrt(2 * N)
    rng = np.random.default_rng(args.seed)              # dynamics/noise stream
    rngp = np.random.default_rng(args.seed + 50000)     # phase stream

    batch = bool(args.batch)
    lh = []
    for _epoch in range(args.num_epochs):
        el = 0.0
        dJ_batch = np.zeros((N, N))
        for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
            a_in = np.zeros(N, complex); a_in[:2] = amp * inp
            a0 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            rec_f = _measure(J, g, kappa, a_in, Tt, Ts, dt, noise_scale,
                             rng, a0=a0)
            afm = measured_fields(rec_f, a_in, kappa,
                                  sphi * rngp.standard_normal(),
                                  args.rotate_record)
            y = (a_in + np.sqrt(kappa) * afm)[out].real
            err = y - tgt / ts
            el += err ** 2 * ts ** 2
            b = ratio * amp / max(abs(err), EPS_ERR)
            da = np.zeros(N, complex); da[out] = -1j * b * err
            rec_p = _measure(J, g, kappa, a_in + da, Tt, Ts, dt, noise_scale,
                             rng, a0=rec_f.a_end.copy())
            apm = measured_fields(rec_p, a_in + da, kappa,
                                  sphi * rngp.standard_normal(),
                                  args.rotate_record)
            dJ = sb_gradient_matrix(afm, apm - afm, b, 'sigma_y')
            if batch:
                dJ_batch += dJ
            else:
                J -= lr * dJ
        if batch:
            J -= lr * dJ_batch / len(XOR_INPUTS)
        lh.append(el / len(XOR_INPUTS))

    clean_loss = evaluate_xor_loss(J, g, kappa, amp, out_idx=out,
                                   target_scale=ts, T_trans=Tt,
                                   T_sample=p['T_sample_clean'],
                                   dt=p['dt_eval'], noise_scale=0.0)
    lh = np.array(lh)
    mode = 'batch' if batch else 'online'
    print(f"seed={args.seed} noise={args.noise} {mode} sigma_phi={sphi:.3f}  "
          f"train_final={lh[-1]:.3e}  clean_eval={clean_loss:.3e}", flush=True)
    fname = os.path.join(
        args.out_dir,
        f'phase_train_{mode}_noise{args.noise}_sphi{sphi:.4f}_seed{args.seed}.npz')
    np.savez(fname, convention='semiclassical_record',
             rotate_record=bool(args.rotate_record),
             seed=args.seed, noise=args.noise, batch=args.batch,
             sigma_phi=sphi, g=g,
             loss_history=lh, clean_eval_loss=clean_loss)
    print(f"Saved  {fname}", flush=True)


if __name__ == '__main__':
    main()
