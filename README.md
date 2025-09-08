# Scattering-Backpropagation
Code for the work presented in "Training nonlinear optical neural networks with Scattering Backpropagation".

[![arXiv](https://img.shields.io/badge/arXiv-2407.04673-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2508.11750)

Abstract:

As deep learning applications continue to deploy increasingly large artificial neural networks, the associated high energy demands are creating a need for alternative neuromorphic approaches. Optics and photonics are particularly compelling platforms as they offer high speeds and energy efficiency. Neuromorphic systems based on nonlinear optics promise high expressivity with a minimal number of parameters. However, so far, there is no efficient and generic physics-based training method allowing us to extract gradients for the most general class of nonlinear optical systems. In this work, we present Scattering Backpropagation, an efficient method for experimentally measuring approximated gradients for nonlinear optical neural networks. Remarkably, our approach does not require a mathematical model of the physical nonlinearity, and only involves two scattering experiments to extract all gradient approximations. The estimation precision depends on the deviation from reciprocity. We successfully apply our method to well-known benchmarks such as XOR and MNIST. Scattering Backpropagation is widely applicable to existing state-of-the-art, scalable platforms, such as optics, microwave, and also extends to other physical platforms such as electrical circuits.

--------------------------------------------------

Please cite as: N. Dal Cin, F. Marquardt, and C. C. Wanjura, Training nonlinear optical neural networks with Scattering Backpropagation, arXiv:2508.11750 (2025).

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
--------------------------------------------------

![image](/figures/figure0.jpg)

--------------------------------------------------

In this repository we include the code for numerically testing the training of a neuromorphic physical system with Scattering Backpropagation. In particular, we consider a nonlinear system of coupled optical modes which evolves toward a steady state, after a probe signal encoding the network input is injected. For further details we refer to the paper and the code. Inside this repo, the code is divided into self-consistent notebooks which were used to produce the numerical data used in the plots included in the several figures (also uploaded here). Most of the files only requires 'numpy', nevertheless a more efficient version of the code using JAX (for which we suggest a GPU) is provided in /Code_fig_3/ which was used for training MNIST on a convolutional-like neuromorphic architecture --- see the Methods in the paper for more details.
