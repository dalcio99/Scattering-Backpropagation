# Code for *Training nonlinear optical neural networks with Scattering Backpropagation*

Nicola Dal Cin, Florian Marquardt, Clara C. Wanjura

[![arXiv](https://img.shields.io/badge/arXiv-2407.04673-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2508.11750)

## Abstract

As deep learning applications continue to deploy increasingly large artificial neural networks, the associated high energy demands are creating a need for alternative neuromorphic approaches. Optics and photonics are particularly compelling platforms as they offer high speeds and energy efficiency. Neuromorphic systems based on nonlinear optics promise high expressivity with a minimal number of parameters. However, so far, there is no efficient and generic physics-based training method allowing us to extract gradients for the most general class of nonlinear optical systems. In this work, we present Scattering Backpropagation, an efficient method for experimentally measuring approximated gradients for nonlinear optical neural networks. Remarkably, our approach does not require a mathematical model of the physical nonlinearity, and only involves two scattering experiments to extract all gradient approximations. The estimation precision depends on the deviation from reciprocity. We successfully apply our method to well-known benchmarks such as XOR and MNIST. Scattering Backpropagation is widely applicable to existing state-of-the-art, scalable platforms, such as optics, microwave, and also extends to other physical platforms such as electrical circuits.


## Description

This repository contains all simulation code and data behind the figures of the paper and its Supplementary Information (SI). 
Each top-level directory is one self-contained study; the table below maps every figure to its directory, generating scripts), and data.

## Environment

All results were produced with CPU-only Python [GPU required only for the (full) MNIST study]:

```
conda env create -f environment.yml
conda activate sb-env
```

(Python 3.11, numpy 2.0, scipy 1.16, matplotlib 3.10, jax/jaxlib 0.7.2 —
JAX is used only for Jacobians in a few studies; `jupyter` for the notebooks.)

The shared physics core for the semiclassical studies lives in
`Noise_analysis/sc_core.py` and `Noise_analysis/gradient_worker_sc.py`; the
workers in `Robustness_analysis/`, `Different_nonlinearities/`, and
`Comparison_generalizedEP_SB/` import them via a relative `sys.path` entry, so
**the directory layout of this repository must be preserved**.

## Figure ↔ code map

| Figure | Directory | Generate data | Make figure | Data in repo? |
|---|---|---|---|---|
| Main Fig. 2a–c (N=3 XOR) | `3modes_XOR/` | `3modes_a_basis_XOR.ipynb` (run all) | same notebook | – (fast: minutes) |
| Main Fig. 2e,f,h + SI Sec. IX (shot noise) | `Noise_analysis/` | `gradient_worker_sc.py` (+ `_3phase`, `_tsamp` variants) via `submit_sc_*.sh` (SLURM array, ~2000 jobs) | `plot_sweep13.py`, `plot_1Pvs2P_sc.py` | Zenodo (≈1.3 GB, see below) |
| Main Fig. 2g + SI Sec. V.B.1 (quasi-reciprocity) | `Quasi_reciprocity_static_study/` | `run_sweep_dual.py` | `plot_paper.py` | `Data/angle_stats_dual.npz` |
| Main Fig. 3 (MNIST) | `MNIST_training/` | `MNIST_training_metrics/conv_MNIST.py` (~days on CPU per g) | from saved metrics in `Data/` | `Data/MNIST_data_g=*.np[yz]` |
| SI Sec. V.B.2 (MNIST quasi-reciprocity) | `MNIST_training/Network_quasi-reciprocity/` | driver in the folder (uses `qs_core.py`) | plot script in the folder | `Data/quasi_symmetry_results.npz` |
| SI Sec. V.B.3 (quasi-rec. during training) | `Quasi-reciprocity_and_gradient_approx_during_training/` | `training_angles_v4.py` (SB_x-trained run) and `training_angles_v4_sigmay.py` (SB_y-trained run), `SB_LR=0.05` | `plot_v4.py <results_dir>` | both `Data/results_v4*/training_angles_v4.npz` |
| SI Sec. VII (EP vs SB) | `Comparison_generalizedEP_SB/` | `ep_vs_sb.py` (gradient fidelity), `ep_vs_sb_training.py` (training; `EPVS_GA0`, `EPVS_MODE`, `EPVS_ARMS` env vars) | `make_ep_vs_sb_paper_figs.py` | `Data/*.npz` |
| SI Sec. VIII (unidirectional two-ring) | `Training_with_unidirectional_light_transmission/` | `unidirectional_transmission_example.ipynb` | same notebook | – (fast: minutes) |
| SI Sec. X.A (self vs cross-Kerr, N=10) | `10modes_XOR_light_quadrature_basis/` | `stats_10modes_...ipynb` (toggle `G` for panel a/b) | same notebook | – (~30 min per panel) |
| SI Sec. X.B (initial-condition robustness, N=3) | `3modes_XOR_different_g_and_IVP/` | `stats_3modes_...ipynb` | same notebook | – (~20 min) |
| SI Sec. X.C (other nonlinearities) | `Different_nonlinearities/` | `gradient_worker_nl_sc.py` via `submit_nl_sc.sh` (24 jobs) | `make_paper_figs.py` | `Data/results_nl_sc/` |
| SI Sec. X.D (disorder, drift, phase, thermal) | `Robustness_analysis/` | `robustness_seed_worker_sc.py`, `drift_worker_sc.py`, `Phase_perport/*` via `submit_*.sh` | plot scripts in the folder | `results_*` directories |

Notes:

- **Batched vs online updates.** The Sec. IX and Sec. X.C studies use batched
  gradients (one update per epoch, `train_fixed_ratio` in `sc_core.py`); the
  Sec. X.D robustness studies deliberately use online (per-pattern) updates,
  implemented in their own workers. `sc_core.train()` is an unused online
  utility and was NOT used for any figure.
- **Symmetric nudging.** The `*_3phase*` workers implement the two-experiment
  (±) linear-response estimate of SI Sec. IV.A.
- **Phase-instability figures** use the per-port results in
  `Robustness_analysis/Phase_perport/results_*_perport_rotrec/`.
- **SLURM scripts** are provided as run recipes; adapt the headers (partition,
  environment activation) to your cluster.


## License and citation

MIT License (see `LICENSE`). If you use this code, data or results, please cite the paper as:

N. Dal Cin, F. Marquardt, and C. C. Wanjura, Training nonlinear optical neural networks with Scattering Backpropagation, arXiv:2508.11750 (2025).

```
@article{dalcin2025ScattBackprop,
  title = {Training nonlinear optical neural networks with Scattering Backpropagation},
  author = {Dal Cin, Nicola and Marquardt, Florian and Wanjura, Clara C},
  journal = {arXiv preprint arXiv:2508.11750},
  year = {2025},
  doi = {arXiv:2508.11750},
  url = {https://arxiv.org/abs/2508.11750}
}
```

(cf.~`CITATION.cff`).
