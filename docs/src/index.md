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


### Lasso and ℓ₁-regularized logistic regression from data

For given data $A \in \mathbb{R}^{m \times n}$ (dense or sparse) and $b \in \mathbb{R}^m$, `model, nls_model = lasso_model(A, b)` returns models of
```math
f(x) = \tfrac{1}{2} \|Ax - b\|_2^2,
```
optionally with the bounds $x \geq 0$ (`bounds = true`), and `model = logistic_model(A, b)` returns a model of
```math
f(x) = \sum_{i=1}^m \log\left(1 + \exp(-b_i a_i^T x)\right),
```
where $a_i^T$ is the $i$-th row of $A$ and $b_i \in \{-1, 1\}$.
Both models implement Hessian-vector products, so they can be used with second-order methods (e.g., `R2N`) with exact Hessians or wrapped in a quasi-Newton model.
The usual regularizer is $h(x) = \lambda \|x\|_1$.

### Test set of Lopes, Santos and Silva (2019)

The 49 Lasso, 49 logistic regression and 12 nonnegative Lasso problems used in

> R. Lopes, S. A. Santos and P. J. S. Silva, *Accelerating block coordinate descent methods with identification strategies*, Computational Optimization and Applications 72, 609–640 (2019). [doi:10.1007/s10589-018-00056-8](https://doi.org/10.1007/s10589-018-00056-8)

are available once the authors' data has been downloaded.
Download `Data-Files.zip` (≈ 3.2 GB) from [the authors' folder](https://drive.google.com/drive/folders/1yUqNvwDazKED-Umfe5O95WP-JyedkLnm) and extract it; it contains the folders `Data-Lasso`, `Data-Logistic` and `Data-Non-Negative-Lasso` of MAT files.
Then
```julia
using MAT, ProximalOperators, RegularizedProblems
ENV["ABCD_DATA_DIR"] = "/path/to/Data-Files"   # folder containing Data-Lasso, ...

abcd_problems(:lasso_test)                      # names of the 31 Lasso problems of the comparisons
model, nls_model, data = abcd_lasso_model("SR10")   # ½‖Ax - b‖², data.λ, data.ftarget
model, data = abcd_logistic_model("SR10")           # file Data-Logistic/SRlog10.mat
reg_nlp, reg_nls, data = setup_abcd_lasso_l1("SC8") # with h = NormL1(data.λ)
reg_nlp, data = setup_abcd_logistic_l1("SC8")
```
Problems `NN1`–`NN12` are nonnegative Lasso problems (the models include the bounds $x \geq 0$).
In each case, `data.λ = 0.1 ‖∇f(0)‖∞` is the regularization parameter used in the paper and `data.ftarget` is the target objective value used there as stopping criterion (an upper bound on the optimal value with relative error of order $10^{-4}$).
See `abcd_problems` for the predefined sets of problems and `abcd_problem_info` for the origin and size of each problem.
