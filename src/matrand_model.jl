export random_matrix_completion_model, MIT_matrix_completion_model

function mat_rand(m::Int, n::Int, r::Int, sr::Float64, va::Float64, vb::Float64, c::Float64)
  xl = rand(Uniform(-0.1, 0.3), m, r)
  xr = rand(Uniform(-0.1, 0.3), n, r)
  xs = xl * xr'
  Ω = findall(<(sr), rand(m, n))
  B = xs[Ω]
  B = (1 - c) * add_gauss(B, va, 0; clip = true) + c * add_gauss(B, vb, 0; clip = true)
  ω = zeros(Int64, size(Ω, 1))   # Vectorize Omega
  for i = 1:size(Ω, 1)
    ω[i] = Ω[i][1] + size(Ω, 2) * (Ω[i][2] - 1)
  end
  return xs, B, ω
end

function matrix_completion_model(xs, B, ω)
  m, n = size(xs)
  res = vec(fill!(similar(xs), 0))

  function resid!(res, x)
    res .= 0
    res[ω] .= x[ω] .- B
    res
  end

  function jprod_resid!(Jv, x, v)
    Jv .= 0
    Jv[ω] .= v[ω]
    Jv
  end

  function obj(x)
    resid!(res, x)
    dot(res, res) / 2
  end

  grad!(r, x) = resid!(r, x)

  x0 = rand(eltype(B), m * n)
  FirstOrderModel(obj, grad!, x0, name = "MATRAND"),
  FirstOrderNLSModel(resid!, jprod_resid!, jprod_resid!, m * n, x0, name = "MATRAND-LS"),
  vec(xs)
end

"""
    model, nls_model, sol = random_matrix_completion_model(; kwargs...)

Return an instance of an `NLPModel` and an instance of an `NLSModel` representing
the same matrix completion problem, i.e., the square linear least-squares objective

   ½ ‖P(X - A)‖²

in the Frobenius norm, where X is the unknown image represented as an m x n matrix,
A is a fixed image, and the operator P only retains a certain subset of pixels of
X and A.

## Keyword Arguments

* `m :: Int`: the number of rows of X and A (default: 100)
* `n :: Int`: the number of columns of X and A (default: 100)
* `r :: Int`: the desired rank of A (default: 5)
* `sr :: AbstractFloat`: a threshold between 0 and 1 used to determine the set of pixels
retained by the operator P (default: 0.8)
* `va :: AbstractFloat`: the variance of a first Gaussian perturbation to be applied to A (default: 1.0e-4)
* `vb :: AbstractFloat`: the variance of a second Gaussian perturbation to be applied to A (default: 1.0e-2)
* `c :: AbstractFloat`: the coefficient of the convex combination of the two Gaussian perturbations (default: 0.2).

## Return Value

An instance of an `NLPModel` and of a `FirstOrderNLSModel` that represent the same
matrix completion problem, and the exact solution.
"""
function random_matrix_completion_model(;
  m::Int = 100,
  n::Int = 100,
  r::Int = 5,
  sr::Float64 = 0.8,
  va::Float64 = 1.0e-4,
  vb::Float64 = 1.0e-2,
  c::Float64 = 0.2,
)
  xs, B, ω = mat_rand(m, n, r, sr, va, vb, c)
  matrix_completion_model(xs, B, ω)
end

function perturb(I, c = 0.8, p = 0.8)
  Ω = findall(<(p), rand(256, 256))
  ω = zeros(Int, size(Ω, 1))   # Vectorize Omega
  for i = 1:size(Ω, 1)
    ω[i] = Ω[i][1] + 256 * (Ω[i][2] - 1)
  end
  X = fill!(similar(I), 0)
  B = I[Ω]
  B = c * add_gauss(B, sqrt(0.001), 0) + (1 - c) * add_gauss(B, sqrt(0.1), 0)
  X[Ω] .= B
  X, B, ω
end

"""
    model, nls_model, sol = MIT_matrix_completion_model()

A special case of matrix completion problem in which the exact image is a noisy
MIT logo.

See the documentation of `random_matrix_completion_model()` for more information.
"""
function MIT_matrix_completion_model()
  I = ones(256, 256)
  I[:, 1:20] .= 0.1
  I[1:126, 40:60] .= 0
  I[:, 80:100] .= 0
  I[1:40, 120:140] .= 0
  I[80:256, 120:140] .= 0.5
  I[1:40, 160:256] .= 0
  I[80:256, 160:180] .= 0

  X, B, ω = perturb(I, 0.8, 0.8)
  matrix_completion_model(X, B, ω)
end
