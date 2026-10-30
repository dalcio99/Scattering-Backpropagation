# MNIST training (main Fig. 3) and quasi-reciprocity check (SI Sec. V.B.2)

Layered network of self-Kerr resonators, (28x28)-(12x12)-(5x5)-10
(kernels 6 and 4, stride 2, dense final layer): N = 963 nodes, 6797 trainable
parameters. NOTE: this study uses kappa'_j = kappa_j = 1 (total damping 2),
unlike the XOR studies where kappa' = 0.

- `Example/` — small self-contained demo of the architecture and SB updates
  (`conv_arch_example.ipynb` + helper modules).
- `MNIST_training_metrics/conv_MNIST.py` — the training driver (SB, sigma_y
  error signal, cross-entropy with softmax temperature 0.1, batch 10,
  eta = 0.1, beta = 0.01, T_max = 60, dt = 0.1). Paper runs: full 60k/10k
  dataset, 400 epochs, g/kbar in {0, 0.1, 0.2, 0.3, 0.4} (days of CPU time per
  g; dataset size can be reduced via SB_MNIST_TRAIN_SIZE / SB_MNIST_TEST_SIZE
  env vars for a quick demo). The paper metrics are shipped in
  `MNIST_training_metrics/Data/MNIST_data_g=*.np[yz]` (test accuracy 92.6%
  linear -> 97.4% at g = 0.2).
- `Network_quasi-reciprocity/` — quasi-reciprocity of the trained-architecture
  network on 50 MNIST test images: driver `quasi_symmetry_angle.py` (uses the
  shipped `J_mnist_seed41.npy` initialization and `Data/mnist_test_*.npy`),
  results `Data/quasi_symmetry_results.npz`, figure
  `plot_quasi_symmetry_paper.py`.
