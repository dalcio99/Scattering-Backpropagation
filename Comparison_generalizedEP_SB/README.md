# Vector-field EP vs Scattering Backpropagation (SI Sec. VII)

Controlled comparison on the standard $N=10$ self-Kerr XOR setting (deterministic).
For the quadratic single-quadrature cost used here, the EP force nudge equals
the SB $\sigma_x$ port nudge, so EP and SB share identical free/nudged experiments
and differ ONLY by one complex conjugation in the classical contraction:
EP pairs $Im[\delta a * \bar{a}^*]$ ($P_1=P_2=I$ in the $(x,p)$ basis), SB pairs
$Im/Re[\delta a * \bar{a}]$ ($P_1=P_2=\sigma_z$ or $\sigma_x$, i.e. $U=\sigma_x$ or $\sigma_y$ in the
$(a,a^*)$ basis).

- `ep_vs_sb.py` — gradient fidelity at initialization: cos(estimate, exact IFT
  gradient) vs nonlinearity (panel a) and vs coupling scale at g=0 (panel b).
  Includes the g=0 closed-form validation of the EP implementation
  (`ep_gradient_analytic_g0`: the exact adjoint computation with Jac in place
  of Jac^T), which agrees with the simulated estimator to cos = 1.000000.
- `ep_vs_sb_training.py` — training comparison; env vars: `EPVS_GA0`
  ($g\langle|a_{in}|^2\rangle_{\mathrm{avg}}/\bar{\kappa}^2$), `EPVS_ARMS` (e.g. `sb_x,sb_y,ep,exact`), `EPVS_MODE`
  (`batched` default / `online`), `EPVS_EPOCHS`, `EPVS_SEEDS`. eta = 5e-4,
  beta = 1e-3, 800 epochs, 8 seeds; runs are hard-stopped if the state diverges.
- `make_ep_vs_sb_paper_figs.py` — the SI figure panels (cos similarity;
  loss histories at ga0 = 0.1 and 5, log-y/linear-x).

Data (`Data/`): `ep_vs_sb_data.npz` (fidelity) and
`ep_vs_sb_training_data_ga{0.1,5}.npz` (the SI training panels).
