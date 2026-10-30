# Unidirectional two-ring example (SI Sec. VIII)

`unidirectional_transmission_example.ipynb`: two ring resonators coupled
sequentially by a waveguide [right/left movers $a$, $b$; inter-ring coupling
$-\sqrt{\kappa_1\kappa_2}$]; trains the two detunings on the +-1 -> +-0.1
regression using the generalized SB theorem with the local symmetry
$U: a_j \to b_j^*$ (error signal injected into b_in at the output ring).
kappa = (0.8, 1.3), g = 0.1, beta = dt = 1e-2, eta = 0.1, 100 epochs.
Runs in minutes.
