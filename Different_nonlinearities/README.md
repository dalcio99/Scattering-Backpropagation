# Different nonlinearities — semiclassical convention

Port of `Corrected_XOR/Other_nonlinearities/` to the sc_core.py physics
(noise $\langle dW dW^\dagger\rangle = dt/2$, record-based estimators). Same grid and parameters
(kerr/saturable/tpa, ga0 = 3/3/5, ratio 0.04, 800 epochs, sigma_y, T_m 20/200).

Cluster files: `../sc_core.py` + `gradient_worker_nl_sc.py` + `submit_nl_sc.sh`
(the worker also finds sc_core.py if placed alongside).

    mkdir -p logs results_nl_sc
    sbatch --array=0-23 submit_nl_sc.sh      # 3 nl x 2 noise x 4 seeds

Old FD caches do NOT apply (record-based loss); recomputed on demand.
Smoke-tested 2026-07-17: cos_init vs per-nl IFT = 0.953 (kerr), 0.983
(saturable), 0.972 (tpa), matching the old study's 0.95-0.98.
