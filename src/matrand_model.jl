export mat_rand, random_matrix_completion_model, random_matrix_completion_eq_model, MIT_matrix_completion_model

function mat_rand(m::Int, n::Int, r::Int, sr::Float64, va::Float64, vb::Float64, c::Float64)
  xl = rand(Uniform(-0.1, 0.3), m, r)
  xr = rand(Uniform(-0.1, 0.3), n, r)
  xs = xl * xr'
  Ω = findall(<(sr), rand(m, n))
  B = xs[Ω]
  B = (1 - c) * add_gauss(B, va, 0; clip = false) + c * add_gauss(B, vb, 0; clip = false)
  ω = zeros(Int64, size(Ω, 1))   # Vectorize Omega
  for i = 1:size(Ω, 1)
    ω[i] = Ω[i][1] + n * (Ω[i][2] - 1)
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

An instance of an `NLPModel` and of an `NLSModel` that represent the same
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

"""
    model, sol = random_matrix_completion_eq_model(; kwargs...)

Return an instance of an `NLPModel` representing the equality-constrained 
matrix completion problem in forward mode, i.e., the linear constraint

`` X_{ij} = A_{ij} \\ \\forall i,j \\in E, ``

where `` E \\subseteq \\mathbb{R}^m \\times \\mathbb{R}^n ``.
Alternatively, one can also represent the equality-constrained 
matrix completion problem in backward mode, i.e, the optimization problem

`` \\min_{\\sigma, U, V} 0 \\quad \\text{s.t.} \\ U^T U = I, \\ V^T V = I, \\ \\sum_k \\sigma_k u_k^i v_k^j = A_{ij}.``

where `` \\sigma \\in \\mathbb{R}^l ``, `` U \\in \\mathbb{R}^m \\times \\mathbb{R}^l `` and 
`` V \\in \\mathbb{R}^l \\times \\mathbb{R}^n `` with `` l \\coloneqq \\min\\{m, n\\} ``.

## Keyword Arguments

* `m :: Int = 100`: the number of rows of X and A;
* `n :: Int = 100`: the number of columns of X and A;
* `r :: Int = 5`: the desired rank of A;
* `sr :: Float64 = 0.8`: a threshold between 0 and 1 used to determine the set of pixels retained by the operator P;
* `mode :: Symbol = :forward`: either `:forward` or `:backward`, represents which form of the equality constrained matrix completion problem is to be represented (see above).

## Return Value

n instance of an `NLPModel` representing the
equality constrained matrix completion problem, and the exact solution.
"""
function random_matrix_completion_eq_model(;
  m::Int = 100,
  n::Int = 100,
  r::Int = 5,
  sr::Float64 = 0.8,
  mode::Symbol = :forward
)
  xs, B, ω = mat_rand(m, n, r, sr, 0.0, 0.0, 0.0)
  return random_matrix_completion_eq_model(xs, B, ω, mode = mode)
end

function random_matrix_completion_eq_model(xs, B, ω; mode = :foward)
  nlp, x0 = mode == :forward ? random_matrix_completion_eq_model_forward(xs, B, ω) :
    random_matrix_completion_eq_model_backward(xs, B, ω)
end

function random_matrix_completion_eq_model_forward(xs, B, ω)

  m, n = size(xs)
  xs = collect(vec(xs))

  # Constrained API
  function cons!(cx, x)
    @views cx .= x[ω] .- B
  end

  function jprod!(jv, x, v)
    @views jv .= v[ω]
  end

  function jtprod!(jtv, x, v)
    jtv .= 0
    @views jtv[ω] .= v
  end

  function hprod!(hv, x, y, v; obj_weight = one(T))
    return hv .= 0
  end

  function hess_coord!(vals, x, y; obj_weight = one(T))
    return zeros(Float64, 0)
  end

  rows_jac = collect(1:length(ω))
  cols_jac = ω

  function jac_coord!(vals, x)
    vals .= 1
  end

  # Unconstrained API
  function obj(x)
    return 0.0
  end

  function grad!(g, x)
    g .= 0
  end

  function hprod!(hv, x, v; obj_weight = 1.0)
    return hv .= 0
  end

  function hess_coord!(vals, x; obj_weight = 1.0)
    return zeros(Float64, 0)
  end

  nlp = NLPModel(
    zero(xs), 
    obj, 
    grad = grad!,
    hprod = hprod!,
    hess_coord = (zeros(Float64, 0), zeros(Float64, 0), hess_coord!),
    cons = (cons!, zero(B), zero(B)),
    jprod = jprod!,
    jtprod = jtprod!,
    jac_coord = (rows_jac, cols_jac, jac_coord!)
  )

  return nlp, xs

end

function random_matrix_completion_eq_model_backward(xs, B, ω)
  m, n = size(xs)
  l = min(m, n)
  xs_svd = svd(xs)
  xs = collect(vcat(vec(xs_svd.U), xs_svd.S, vec(xs_svd.Vt)))

  # Constrained API
  function c!(cx, x)
    U = @views reshape(x[1:m*l], m, l)
    σ = @views x[m*l + 1 : m*l + l]
    V = @views reshape(x[m*l + l + 1:end], n, l)

    n_entries_l = Int(l*(l+1)/2)

    @inbounds for i = 1:l
      @inbounds for j = 1:i
        cx_idx = Int(j + i*(i-1)/2)
        if i == j
          @views cx[cx_idx] = dot(U[:, i], U[:, j]) - 1
          @views cx[n_entries_l + cx_idx] = dot(V[i, :], V[j, :]) - 1
        else
          @views cx[cx_idx] = dot(U[:, i], U[:, j])
          @views cx[n_entries_l + cx_idx] = dot(V[i, :], V[j, :])
        end
      end
    end

    offset = 2*n_entries_l
    @inbounds for i in eachindex(ω)
      i_idx = mod(ω[i], n)
      i_idx = i_idx == 0 ? n : i_idx
      j_idx = Int((ω[i] - i_idx)/n) + 1
      cx[offset + i] = - B[i]
      @inbounds for j = 1:l
        cx[offset + i] += σ[j]*U[i_idx, j]*V[j, j_idx]
      end
    end    
  end


  # Unconstrained API
  function obj(x)
    return 0.0
  end
  x0 = zeros(Float64, m*l + n*l + l)
  ncon = l*(l+1) + length(B)
  return ADNLPModel!(
    obj,
    x0,
    c!,
    zeros(ncon),
    zeros(ncon)
  ), xs
end

