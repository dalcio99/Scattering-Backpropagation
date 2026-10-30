# N=10 XOR: self- vs cross-Kerr (SI Sec. X.A)

`stats_10modes_quadrature_basis_XOR_self_VS_cross-Kerr.ipynb`: ten training runs
(different J inits) of the fully connected N=10 network with g = 0.3, using the
SB sigma_z update in the quadrature basis, which corresponds to U=sigma_x in the (a,a*) basis. 
(T_max = 60, dt = 0.1, beta = 0.01, eta = 1e-3, 1000 epochs; output = 10*x_out at node 5). 
Toggle the `G` block in the first cell to switch between the nearest-neighbour cross-Kerr
ring (as stored, SI Fig. 4b) and self-Kerr (`G = np.eye(N)`, SI Fig. 4a); the seed
stream is identical for both. ~30 min per configuration.

