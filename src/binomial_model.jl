# binomial_model.jl
using LinearAlgebra

export binomial_model

"""
    nlp = binomial_model(A, b)

Return an instance of an `NLPModel` representing the binomial logistic regression problem.

Minimize

    f(x) = sum( log(1 + exp(a_i' * x)) - b_i * (a_i' * x) )

where b_i ∈ {0, 1} and `A` is (m features) × (n samples).
Equivalently, z = A' * x (length n).
"""
function binomial_model(A, b)
  m, n = size(A)  # m features, n samples

  # Basic sanity checks (cheap, helps catch silent dimension issues)
  length(b) == n || throw(DimensionMismatch("length(b) = $(length(b)) must equal number of samples n = $n"))
  eltype(A) <: Real || throw(ArgumentError("A must have a real element type"))
  eltype(b) <: Real || throw(ArgumentError("b must have a real element type"))

  # Pre-allocate buffers
  Ax    = zeros(Float64, n)  # z = A' * x
  p     = zeros(Float64, n)  # sigmoid(z)
  w     = zeros(Float64, n)  # p*(1-p)
  tmp_n = zeros(Float64, n)  # sample-space buffer
  tmp_v = zeros(Float64, n)  # sample-space buffer

  # Numerically stable logistic sigmoid
  @inline function sigmoid(t::Float64)
    if t ≥ 0.0
      return 1.0 / (1.0 + exp(-t))
    else
      et = exp(t)
      return et / (1.0 + et)
    end
  end

  # Numerically stable softplus: log(1 + exp(t))
  @inline function softplus(t::Float64)
    t > 0.0 ? (t + log1p(exp(-t))) : log1p(exp(t))
  end

  # Objective: sum(softplus(A'x) - b .* (A'x))
  function obj(x)
    mul!(Ax, A', x)  # Ax = A' * x
    s = 0.0
    @inbounds @simd for i in 1:n
      zi = Ax[i]
      s += softplus(zi) - Float64(b[i]) * zi
    end
    return s
  end

  # Gradient: A * (sigmoid(A'x) - b)
  function grad!(g, x)
    mul!(Ax, A', x)

    @inbounds @simd for i in 1:n
      p[i] = sigmoid(Ax[i])
      tmp_n[i] = p[i] - Float64(b[i])
    end

    mul!(g, A, tmp_n)  # g = A * (p - b)
    return g
  end

  # Hessian-vector product: hv = A * ( (p .* (1-p)) .* (A'v) )
  #
  # IMPORTANT: NLPModels-style in-place signature is typically hprod!(hv, x, v; obj_weight=...)
  function hprod!(hv, x, v; obj_weight = 1.0)
    mul!(Ax, A', x)

    @inbounds @simd for i in 1:n
      pi = sigmoid(Ax[i])
      w[i] = pi * (1.0 - pi)
    end

    mul!(tmp_v, A', v)   # tmp_v = A' * v
    @. tmp_v *= w        # tmp_v .= w .* tmp_v
    mul!(hv, A, tmp_v)   # hv = A * tmp_v

    if obj_weight != 1.0
      rmul!(hv, obj_weight)
    end
    return hv
  end

  x0 = zeros(Float64, m)

  # Construct an NLPModel with registered gradient + Hessian-vector product
  return ManualNLPModels.NLPModel(
    x0,
    obj;
    grad = grad!,
    hprod = hprod!,
    meta_args = Dict(:name => "Binomial"),
  )
end
