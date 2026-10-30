# N=3 XOR: nonlinearity comparison and initial-condition robustness (SI Sec. X.B)

`stats_3modes_a_basis_XOR_different_g_and_IVP.ipynb`: trains the N=3 network for
g in {0, 0.1, 0.2} (T_max = 60, dt = 0.01, beta = eta = 1e-3, 200 epochs; a
separate run from main Fig. 2a-c) and evaluates the trained networks from 100
random initial conditions (SI Fig. 5b box plots; SI Fig. 6 weight maps).
The outputs produced in the notebook correspond to the published figures.
