# RegularizedProblems.jl

[![CI](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl/actions/workflows/ci.yml)
[![](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/RegularizedProblems.jl/dev)

## Synopsis

`RegularizedProblems` is a repository of optimization problems implemented in pure Julia.
Contrary to what the name suggests, the problems are *not* regularized but they *should be*.
However, the choice of regularizer is left to the user.

The problems concerned by the package have the form

<p align="center">
minimize f(x) + h(x)
</p>

where f: ℝⁿ → ℝ has Lipschitz-continuous gradient and h: ℝⁿ → ℝ is lower semi-continuous and proper.
The smooth term f describes the objective to minimize while the role of the regularizer h is to select
a solution with desirable properties: minimum norm, sparsity below a certain level, maximum sparsity, etc.

This repository gives access to several f terms.
Regularizers h should be taken from [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl).

## How to Install

Until this package is registered, use
```julia
pkg> add https://github.com/optimizers/RegularizedProblems.jl
```

## What is Implemented?

Please refer to the documentation.

## Related Software

* [ShiftedProximalOperators.jl](https://github.com/rjbaraldi/ShiftedProximalOperators)
* [RegularizedOptimization.jl](https://github.com/UW-AMO/RegularizedOptimization.jl)

## References

* A. Y. Aravkin, R. Baraldi and D. Orban, *A Proximal Quasi-Newton Trust-Region Method for Nonsmooth Regularized Optimization*, Cahier du GERAD G-2021-12, GERAD, Montréal, Canada. https://arxiv.org/abs/2103.15993

