# Static quasi-reciprocity study (main Fig. 2g; SI Sec. V.B.1)

Steady-state (fsolve-based, no time integration) study of the quasi-symmetry
$cos(S^\dagger, U S U^{-1})$ and the SB gradient approximation vs system size N,
nonlinearity g, initialization variance (1 or 1/N), and nonlinearity type
(self-Kerr, nearest-neighbour cross-Kerr ring, all-to-all cross-Kerr).

- `run_sweep_dual.py` — the sweep (30 seeds per point; exact IFT gradients,
  beta->0 limit). Writes `Data/angle_stats_dual.npz`.
- `plot_paper.py` — the published figures (scatter_cosine_faded,
  heatmap_convergence, heatmap_Gz_cosine, dist_cosGz_vs_g violins with
  N = 20/30/40, cos_Sy_selfkerr_1overN) from `Data/angle_stats_dual.npz`.

Note: the code works in the (x,p) quadrature basis; quadrature sigma_z/sigma_x
correspond to sigma_x/sigma_y in the (a,a*) basis of the paper.
