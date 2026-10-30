"""
cos_phase_perport_worker.py — gradient degradation under INDEPENDENT-PER-PORT
readout phase noise, at INIT; semiclassical convention (sc_core.py physics,
record-based estimators).

For one J0 seed, compute cos(SB gradient, reference gradient) at the initial
network vs sigma_phi (clean & noisy dynamics). References (uniform sc policy):
  cos_clean : vs the noiseless exact gradient (deterministic central FD, eps=1e-3);
  cos_noisy : vs the noisy FD-CRN gradient of the record-based loss
              (T_m^FD = 400, eps = 1e-4, common random numbers, PHASE-FREE) —
              this reference is a property of the network, independent of the
              readout-phase model, so it is identical to the common-phase run;
              cached per seed as fd_grad_*.npy in the output directory.

The SB gradient uses the field estimate at all N ports, so
this is exactly where independent per-port phase degrades the alignment.

Phase-error model as in phase_train_perport_worker.py: default rotates each
port's estimator a_hat_n; --rotate_record rotates the full record.

npz schema + file naming match ../cos_phase_worker_sc.py so
../../../robustness/plot_cos_phase.py works via --results_dir; the extra field
phase_model='independent_per_port' distinguishes the runs.

Cluster deps (flat copy, all in one dir): sc_core.py, robustness_seed_worker_sc.py,
gradient_worker_sc.py, this file.

Usage
-----
  python cos_phase_perport_worker.py --seed 0 --n_phi 20 --n_real 20 \
      --out_dir results_cos_phase_perport
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
from robustness_seed_worker_sc import DEFAULTS, EPS_ERR, _measure
from gradient_worker_sc import (fd_gradient_crn, T_SAMPLE_FD, EPS_FD,
                                T_TRANS as SWEEP_T_TRANS)


def measured_fields_perport(rec, a_in, kappa, phi_vec, rotate_record):
    """Independent per-port readout phase: phi_vec has one entry per mode."""
    if rotate_record:
        rot_out = rec.a_out_avg * np.exp(1j * phi_vec)
        return (rot_out - a_in) / np.sqrt(kappa)
    return rec.a_hat * np.exp(1j * phi_vec)


def _cos(A, B):
    a, b = A.ravel(), B.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--g', type=float, default=0.003125)
    ap.add_argument('--n_phi', type=int, default=20)
    ap.add_argument('--n_real', type=int, default=20)
    ap.add_argument('--phi_max', type=float, default=np.pi / 4)
    ap.add_argument('--eps_fd', type=float, default=1e-3)
    ap.add_argument('--rotate_record', action='store_true')
    ap.add_argument('--out_dir', default=os.path.join(
        HERE, 'results_cos_phase_perport'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    p = dict(DEFAULTS); N = p['N']; out = p['out_idx']; ts = p['target_scale']
    kappa = np.ones(N) * p['kappa_val']; amp = np.sqrt(p['a0_in_squared'] * N)
    g = args.g; Tt = p['T_trans']; ratio = p['ratio']; dt = p['dt']
    Tc, Tn = p['T_sample_clean'], p['T_sample_noisy']

    rng0 = np.random.default_rng(args.seed)
    R = rng0.standard_normal((N, N)); J = (R + R.T) / np.sqrt(2 * N)

    # exact (FD central-difference) gradient of the noiseless batch loss
    def loss(Jm):
        tot = 0.0
        for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
            a_in = np.zeros(N, complex); a_in[:2] = amp * inp
            rec = _measure(Jm, g, kappa, a_in, Tt, Tc, dt, 0.0,
                           np.random.default_rng(0))
            y = rec.a_out_avg[out].real
            tot += (y - tgt / ts) ** 2 * ts ** 2
        return tot / len(XOR_INPUTS)

    Gtrue = np.zeros((N, N)); e = args.eps_fd
    for i in range(N):
        for j in range(i, N):
            Jp = J.copy(); Jp[i, j] += e
            Jm_ = J.copy(); Jm_[i, j] -= e
            if i != j:
                Jp[j, i] += e; Jm_[j, i] -= e
            d = (loss(Jp) - loss(Jm_)) / (2 * e)
            Gtrue[i, j] = d; Gtrue[j, i] = d

    # noisy reference: FD-CRN of the record-based loss (sweep convention),
    # phase-free, cached per seed (identical to the common-phase run)
    fd_tag = os.path.join(args.out_dir,
                          f'fd_grad_g{g:.6g}_noise1.0_seed{args.seed}.npy')
    if os.path.exists(fd_tag):
        Gtrue_noisy = np.load(fd_tag)
    else:
        print("  computing FD-CRN noisy reference (T_FD = "
              f"{T_SAMPLE_FD:g}) ...", flush=True)
        Gtrue_noisy = fd_gradient_crn(J, g, kappa, 1.0, SWEEP_T_TRANS,
                                      T_SAMPLE_FD, dt, EPS_FD, args.seed)
        np.save(fd_tag, Gtrue_noisy)

    def sb_grad(sphi, noise_scale, Ts, rng):
        dJ = np.zeros((N, N))
        for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
            a_in = np.zeros(N, complex); a_in[:2] = amp * inp
            a0 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            rec_f = _measure(J, g, kappa, a_in, Tt, Ts, dt, noise_scale,
                             rng, a0=a0)
            afm = measured_fields_perport(rec_f, a_in, kappa,
                                          sphi * rng.standard_normal(N),
                                          args.rotate_record)
            y = (a_in + np.sqrt(kappa) * afm)[out].real
            err = y - tgt / ts
            b = ratio * amp / max(abs(err), EPS_ERR)
            da = np.zeros(N, complex); da[out] = -1j * b * err
            rec_p = _measure(J, g, kappa, a_in + da, Tt, Ts, dt, noise_scale,
                             rng, a0=rec_f.a_end.copy())
            apm = measured_fields_perport(rec_p, a_in + da, kappa,
                                          sphi * rng.standard_normal(N),
                                          args.rotate_record)
            dJ += sb_gradient_matrix(afm, apm - afm, b, 'sigma_y')
        return dJ

    phi_v = np.logspace(-3, np.log10(args.phi_max), args.n_phi)
    cos_clean = np.zeros((args.n_phi, args.n_real))
    cos_noisy = np.zeros((args.n_phi, args.n_real))
    for ip, sphi in enumerate(phi_v):
        for k in range(args.n_real):
            rc = np.random.default_rng(1000 + ip * 100 + k)
            rn = np.random.default_rng(5000 + ip * 100 + k)
            cos_clean[ip, k] = _cos(sb_grad(sphi, 0.0, Tc, rc), Gtrue)
            cos_noisy[ip, k] = _cos(sb_grad(sphi, 1.0, Tn, rn), Gtrue_noisy)
        print(f"  sigma_phi={sphi:.4f}  cos_clean={cos_clean[ip].mean():+.3f}  "
              f"cos_noisy={cos_noisy[ip].mean():+.3f}", flush=True)

    fname = os.path.join(args.out_dir, f'cos_phase_seed{args.seed}.npz')
    np.savez(fname, convention='semiclassical_record',
             phase_model='independent_per_port',
             rotate_record=bool(args.rotate_record),
             ref_clean='det-central-FD', ref_noisy='FD-CRN_TFD400',
             seed=args.seed, g=g, phi_v=phi_v,
             cos_clean=cos_clean, cos_noisy=cos_noisy)
    print(f"Saved  {fname}", flush=True)


if __name__ == '__main__':
    main()
