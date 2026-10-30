# Shot-noise study (main Fig. 2e,f,h; SI Sec. IX)

Stochastic (truncated-Wigner) training of an N=10 self-Kerr XOR network.
Also hosts the shared physics core used by several other studies:

- `sc_core.py` — dynamics (RK4 drift + Euler–Maruyama noise), record-based
  estimators, SB gradient contraction, batched trainer `train_fixed_ratio`
  (the function `train()` is an unused online variant, NOT used in the paper).
- `gradient_worker_sc.py` — one (ga0, nudge-ratio, noise, seed, U) sweep point:
  800-epoch batched training + epoch-1 cosine vs the exact (IFT) gradient;
  FD-CRN reference in the noisy case.
- `gradient_worker_sc_3phase.py` — same with symmetric ($\pm$) nudging (SI Sec. IV.A).
- `gradient_worker_sc_tsamp*.py` — measurement-time (T_m) study.
- `submit_sc_*.sh` — SLURM array recipes for the published sweeps
  (sweep = single nudge, sweep_3phase = symmetric nudging; 8 seeds via
  SEED_OFFSET). Adapt headers to your cluster.
- `plot_sweep.py` — all `fig_s_*` heatmaps/curves (Fig. 2e,f,h and the
  SI `sweep_*`/`sweep_*` figures); `--results_dir Data/results_sc_sweep`
  or `Data/results_sc_sweep_3phase`.
- `plot_1Pvs2P_sc.py` — the T_m comparison figure (SI fig. on measurement time).

Data: `Data/results_sc_sweep`, `Data/results_sc_sweep_3phase`,
`Data/results_tsamp_sc{,_3phase}` (~1.3 GB total; archived on Zenodo — see the
top-level README).

Conventions: input |a0_in| = sqrt(160·N) on modes 1,2; output node 3;
J off-diagonal variance 1/N (diagonal/detunings 2/N); batched updates; fixed
nudge ratio |delta a_in|/|a0_in| with adaptive beta; eta = 5e-4.
