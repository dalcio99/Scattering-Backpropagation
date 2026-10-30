"""
phase_train_perport_worker.py — train under INDEPENDENT-PER-PORT readout phase
noise, semiclassical convention (sc_core.py physics, record-based estimators).

Companion / contrast to ../phase_train_worker_sc.py, which applies a SINGLE
COMMON phase phi to every collection port (an idealization justified only by
common-path / shared-LO referencing). Here every port n = 0..N-1 carries its
OWN independent homodyne phase error

        phi_n ~ N(0, sigma_phi^2),   i.i.d. across ports,

redrawn independently for every steady-state measurement (free and nudged
alike). This is the physically honest model for physically SEPARATED, indepen-
dently referenced output ports (no shared LO / no active phase locking between
ports).

Only the phase draw differs from the common-phase worker: `sigma_phi *
standard_normal(N)` (a vector) instead of `... standard_normal()` (a scalar).
`np.exp(1j*phi)` then rotates each port's estimator by its own phase.

Where the per-port structure matters:
  * the network OUTPUT y = Re[(a_in + sqrt(kappa) a_hat)[out_idx]] reads only the
    (undriven) output node, so the XOR ERROR signal sees only phi_{out};
  * the SB GRADIENT uses the field estimate at ALL ports, so it is exposed to
    the full independent-phase vector — this is where dropping the common-phase
    assumption bites.

Phase-error model (per-port):
  default          : a_hat_n -> a_hat_n e^{i phi_n}            (rotate each estimator)
  --rotate_record  : a_hat_n = (e^{i phi_n} <a_out,n>_Tm - abar_in,n)/sqrt(kappa)
                     (rotates the full record; leaks reflected drive at driven
                     nodes — physically stricter). y at the undriven output node
                     is identical in both models.

Saves the (phase-corrupted) training-loss history and the final CLEAN phase-free
eval loss. npz schema + file naming match ../phase_train_worker_sc.py so the
existing ../../../robustness/plot_phase_train.py works via --results_dir; the
extra field phase_model='independent_per_port' distinguishes the runs.

Grid (identical to the common-phase run, for a direct comparison):
  6 sigma_phi x 2 dynamics (clean/noisy) x 2 modes (online/batch) x 5 seeds.

Cluster deps (flat copy, all in one dir): sc_core.py, robustness_seed_worker_sc.py,
this file. (Structured copy under Robustness_sc/Phase_perport/ also works — the
parent/grandparent dirs are added to sys.path below.)

Usage
-----
  python phase_train_perport_worker.py --seed 0 --sigma_phi 0.1 --noise 1 \
      --num_epochs 800 --out_dir results_phase_train_perport
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE)),
           os.path.join(os.path.dirname(os.path.dirname(HERE)), 'Noise_analysis')):  # shared modules live in Code/Noise_analysis
    if p_ not in sys.path:
        sys.path.insert(0, p_)
from sc_core import sb_gradient_matrix, XOR_INPUTS, XOR_TARGETS
from robustness_seed_worker_sc import (DEFAULTS, EPS_ERR, _measure,
                                       evaluate_xor_loss)


def measured_fields_perport(rec, a_in, kappa, phi_vec, rotate_record):
    """Apply an INDEPENDENT per-port readout phase.

    phi_vec : length-N real array, phi_vec[n] = phase error at port n.
    Returns the phase-corrupted field estimator a_hat_meas (length N).
    """
    if rotate_record:
        rot_out = rec.a_out_avg * np.exp(1j * phi_vec)
        return (rot_out - a_in) / np.sqrt(kappa)
    return rec.a_hat * np.exp(1j * phi_vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--sigma_phi', type=float, required=True,
                    help='per-port readout phase noise std [rad]')
    ap.add_argument('--noise', type=int, default=1, choices=[0, 1])
    ap.add_argument('--batch', type=int, default=0, choices=[0, 1],
                    help='0 online / 1 batch (mean gradient per epoch)')
    ap.add_argument('--rotate_record', action='store_true',
                    help='rotate the full record (incl. reflected drive) '
                         'instead of the field estimator only')
    ap.add_argument('--num_epochs', type=int, default=800)
    ap.add_argument('--g', type=float, default=0.003125)
    ap.add_argument('--out_dir', default=os.path.join(
        HERE, 'results_phase_train_perport'))
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
            afm = measured_fields_perport(rec_f, a_in, kappa,
                                          sphi * rngp.standard_normal(N),
                                          args.rotate_record)
            y = (a_in + np.sqrt(kappa) * afm)[out].real
            err = y - tgt / ts
            el += err ** 2 * ts ** 2
            b = ratio * amp / max(abs(err), EPS_ERR)
            da = np.zeros(N, complex); da[out] = -1j * b * err
            rec_p = _measure(J, g, kappa, a_in + da, Tt, Ts, dt, noise_scale,
                             rng, a0=rec_f.a_end.copy())
            apm = measured_fields_perport(rec_p, a_in + da, kappa,
                                          sphi * rngp.standard_normal(N),
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
    print(f"[perport] seed={args.seed} noise={args.noise} {mode} "
          f"sigma_phi={sphi:.3f}  train_final={lh[-1]:.3e}  "
          f"clean_eval={clean_loss:.3e}", flush=True)
    fname = os.path.join(
        args.out_dir,
        f'phase_train_{mode}_noise{args.noise}_sphi{sphi:.4f}_seed{args.seed}.npz')
    np.savez(fname, convention='semiclassical_record',
             phase_model='independent_per_port',
             rotate_record=bool(args.rotate_record),
             seed=args.seed, noise=args.noise, batch=args.batch,
             sigma_phi=sphi, g=g,
             loss_history=lh, clean_eval_loss=clean_loss)
    print(f"Saved  {fname}", flush=True)


if __name__ == '__main__':
    main()
