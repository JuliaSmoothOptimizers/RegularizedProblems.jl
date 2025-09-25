```@meta
CurrentModule = RegularizedProblems
```

# RegularizedProblems

## Synopsis

This package provides sameple problems suitable for developing and testing first and second-order methods
for regularized optimization, i.e., they have the general form

```math
\min_{x \in \mathbb{R}^n} \ f(x) + h(x),
```

where $f: \mathbb{R}^n \to \mathbb{R}$ has Lipschitz-continuous gradient and $h: \mathbb{R}^n \to \mathbb{R} \cup \{\infty\}$ is lower semi-continuous and proper.
The smooth term f describes the objective to minimize while the role of the regularizer h is to select
a solution with desirable properties: minimum norm, sparsity below a certain level, maximum sparsity, etc.

Models for f are instances of [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) and often represent nonlinear least-squares residuals, i.e., $f(x) = \tfrac{1}{2} \|F(x)\|_2^2$ where $F: \mathbb{R}^n \to \mathbb{R}^m$.

The regularizer $h$ should be obtained from [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl).

The final regularized problem is intended to be solved by way of solver for nonsmooth
regularized optimization such as those in [RegularizedOptimization.jl](https://github.com/UW-AMO/RegularizedOptimization.jl).

## Problems implemented

### Basis-pursuit denoise

Calling `model = bpdn_model()` returns a model representing the smooth underdetermined linear least-squares residual

```math
f(x) = \tfrac{1}{2} \|Ax - b\|_2^2,
```

where $A$ has orthonormal rows.
The right-hand side is generated as $b = A x_{\star} + \varepsilon$ where $x_{\star}$ is a sparse vector, $\varepsilon \sim \mathcal{N}(0, \sigma)$ and $\sigma \in (0, 1)$ is a fixed noise level.

When solving the basis-pursuit denoise problem, the goal is to recover $x \approx x_{\star}$.
In particular, $x$ should have the same sparsity pattern as $x_{\star}$.
That is typically accomplished by choosing a regularizer of the form

* ``h(x) = \lambda \|x\|_1`` for a well-chosen ``\lambda > 0``;
* ``h(x) = \|x\|_0``;
* ``h(x) = \chi(x; k \mathbb{B}_0)`` for ``k \approx \|x_{\star}\|_0``;

where $\chi(x; k \mathbb{B}_0)$ is the indicator of the $\ell_0$-pseudonorm ball of radius $k$.

Calling `model = bpdn_nls_model()` returns the same problem modeled explicitly as a least-squares problem.

### Fitzhugh-Nagumo data-fitting problem

If `ADNLPModels` and `DifferentialEquations` have been imported, `model = fh_model()` returns a model representing the over-determined nonlinear least-squares residual

```math
f(x) = \tfrac{1}{2} \|F(x)\|_2^2,
```

where $F: \mathbb{R}^5 \to \mathbb{R}^{202}$ represents the residual between a simulation of the [Fitzhugh-Nagumo system](https://en.wikipedia.org/wiki/FitzHugh–Nagumo_model) with parameters $x$ and a simulation of the [Van der Pol oscillator](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator) with preset, but unknown, parameters $x_{\star}$.

A feature of the Fitzhugh-Nagumo model is that it reduces to the Van der Pol oscillator when certain parameters are set to zero.
Thus here again, the objective is to recover a sparse solution to the data-fitting problem.
Hence, typical regularizers are the same as those used for the basis-pursuit denoise problem.

## Contributors

```@raw html
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
```
