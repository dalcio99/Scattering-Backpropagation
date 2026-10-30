# Quasi-reciprocity and SB gradient approximation during training (SI Sec. V.B.3)

Study: $N=10$ fully connected networks trained on XOR, tracking the quasi-symmetry
$\cos(S^\dagger,USU^{-1})$ and the alignment $\cos(\nabla C, \nabla^{SB}C)$ along the training trajectory,
for self-Kerr, nearest-neighbour cross-Kerr (ring), and all-to-all cross-Kerr
nonlinearities.

## Scripts

- `training_angles_v4.py` — main run trained with the **SB_x** rule; 
  also runs the method comparison (SB_x / SB_y / exact gradient from the same J initializations).
  Run as: `SB_LR=0.05 python training_angles_v4.py`
- `training_angles_v4_sigmay.py` — identical parameters (same seeds from
  `np.random.seed(777)`, β=0.01, 1000 epochs), but the main run is trained with
  the **SB_y** rule. The method comparison is
  skipped by default (`SB_RUN_CMP=1` re-enables it; `cmp_loss` is then all-NaN
  in the npz). Run as: `SB_LR=0.05 python training_angles_v4_sigmay.py`
- `plot_v4.py` — produces the paper figures from a results directory:
  `python plot_v4.py <results_dir>`

Note on bases: the code works in $(x,p)$ quadratures; code $\sigma_z \leftrightarrow \sigma_x$ in the paper and
code $\sigma_x \leftrightarrow \sigma_y$ in the paper (see the header of `training_angles_v4.py`).

## Data → SI figures

- `Data/results_v4_lr0.05/` — SB_x-trained run (`SB_LR=0.05`), incl. method
  comparison. Produces:
  - `training_v4_fig1_sigmax.pdf`  → SI figure ($\sigma_x$ feedback)
  - `training_v4_fig2_method.pdf`  → SI figure (method comparison)
  - (its `training_v4_fig1_sigmay.pdf` is NOT used in the SI: it would show
    $\sigma_y$ quantities monitored along the SB_x-trained trajectory)
- `Data/results_v4_sigmay_lr0.05/` — SB_y-trained run (`SB_LR=0.05`). Produces:
  - `training_v4_fig1_sigmay.pdf`  → SI figure ($\sigma_y$ feedback; genuinely trained
    with $\nabla^{SB_y}C$)
  - (its `training_v4_fig1_sigmax.pdf` is NOT used in the SI, for the mirrored
    reason; fig2 is skipped since the method comparison was not rerun)

To regenerate the SI figures:

```
python plot_v4.py Data/results_v4_lr0.05          # fig1_sigmax, fig2_method
python plot_v4.py Data/results_v4_sigmay_lr0.05   # fig1_sigmay
```
