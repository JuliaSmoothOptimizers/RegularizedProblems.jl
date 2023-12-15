export qp_rand_model
using .QuadraticModels

"""
    model, x0 = qp_rand_model(n, dens = 1.0e-3)

Return an instance of a `QuadraticModel` representing

   ½ xᵀQx + cᵀx   s.t.  x ≥ 1,

where Q = A + A' + I where A is a random square matrix with density `dens`.

## Arguments

* `n :: Int`: size of the problem,
* `dens :: Real`: density of A used to generate the quadratic model.

## Return Value

An instance of a `QuadraticModel`
"""
function qp_rand_model(n::Int, dens::R = 1.0e-3) where {R <: Real}
  A = sprand(R, n, n, dens)
  H = A + A' + I
  c = randn(R, n)
  l = .-ones(R, n) #-one(R) .- rand(R, n)
  # u = one(R) .+ rand(R, n)
  qp = QuadraticModel(c, H; lvar = l)
  x0 = zeros(R, n)
  qp, x0
end
