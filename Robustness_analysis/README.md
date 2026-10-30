# Robustness study: static disorder, drift, phase instability, thermal noise (SI Sec. X.D)

N = 10 all-to-all self-Kerr XOR networks at the operating point
$g\langle|a_{in}|^2\rangle_{avg}/\bar{\kappa}^2 = 0.5$ (g = 1/320, |a0_in| = 40), trained with ONLINE
(per-pattern) sigma_y updates, eta = 5e-4, nudge ratio 0.04, 800 epochs,
5 seeds; deterministic (T_m = 20) and shot-noise (T_m = 200) variants.
Shared physics core imported from ../Noise_analysis.

Workers (run via the submit_*.sh recipes; adapt SLURM headers):
- `robustness_seed_worker_sc.py` — trains the 5 seed networks, then sweeps the
  six static-disorder channels (detuning, coupling, loss, nonlinearity,
  cross-talk, thermal noise; 15 log-spaced strengths x 15 samples) and the
  static phase-readout sweep. `--a0_in_sq 1600 --g 3.125e-4` reproduces the
  SI thermal-noise row (results_thermal_noise_sc/thermal_a0_1600_5seed_data.npz).
- `drift_worker_sc.py` — common-mode detuning random walk during training
  (200 epochs, 15 sigma_step values x 10 runs x {online-clean, online-noisy,
  frozen} arms), starting from the trained networks.
- `cos_phase_worker_sc.py`, `phase_train_worker_sc.py` and
  `Phase_perport/*_perport_worker.py` — phase-instability studies. The
  PUBLISHED phase figures use the per-port, rotate-record results in
  `Phase_perport/results_*_perport_rotrec/`; the top-level `results_*_sc`
  phase dirs are the common-phase variant.

Plot scripts (defaults point at the packaged results_* dirs):
- `plot_robustness_seeds.py` — training curves + static phase evaluation
  (robustness_training_5seed, robustness_phase_5seed).
- `plot_robustness_summary_withthermal.py` — the six-panel static-disorder
  summary incl. the thermal panel and the threshold table (MSE = 1/16).
- `plot_drift_sweep.py`, `plot_drift_during_training.py` — drift figures
  (the published single-run figure uses sigma_step = 0.009703, the default).
- `plot_phase_train.py`, `plot_cos_phase.py` — phase-instability figures.
