export qp_rand_model
using .QuadraticModels

"""
    model, x0 = qp_rand_model(n; dens = 1.0e-3, convex = false)

Return an instance of a `QuadraticModel` representing

   ½ xᵀHx + cᵀx   s.t.  l ≤ x ≤ u,

with H = A + A' or H = A * A' + I (see the `convex` keyword argument) where A is a random square matrix with density `dens`, `l = -e -tₗ` and `u = e + tᵤ` where `tₗ` and `tᵤ` are sampled from a uniform distribution between 0 and 1.    

## Arguments

* `n :: Int`: size of the problem,

## Keyword arguments

* `dens :: Real`: density of `A`` used to generate the quadratic model (default: `1.0e-3`).
* `convex :: Bool`: true to generate a convex `H` (default: `false`).

## Return Value

An instance of a `QuadraticModel`.
"""
function qp_rand_model(n::Int; dens::R = 1.0e-4, convex::Bool = false) where {R <: Real}
  A = sprandn(R, n, n, dens)
  H = convex ? (A * A') : (A + A') #+ I
  c = randn(R, n)
  l = -one(R) .- rand(R, n)
  u = one(R) .+ rand(R, n)
  qp = QuadraticModel(c, H; lvar = l, uvar = u)
  x0 = zeros(R, n)
  qp, x0
end
