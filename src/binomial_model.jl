# binomial_model.jl
using LinearAlgebra

export binomial_model

"""
    nlp = binomial_model(A, b)

Return an instance of an `NLPModel` representing the binomial logistic regression problem.

Minimize

    f(x) = sum( log(1 + exp(aᵢᵀ x)) - bᵢ (aᵢᵀ x) )

where `bᵢ ∈ {0, 1}`, `A` is `m × n` (`m` features, `n` samples), and `aᵢ` denotes column `i` of `A`.
"""
function binomial_model(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: Real}
  m, n = size(A)  # m features, n samples

  length(b) == n || throw(DimensionMismatch("length(b) = $(length(b)) must equal number of samples n = $n"))

  # Pre-allocate buffers
  Ax    = similar(b)  # z = A' * x
  p     = similar(b)  # sigmoid(z)
  w     = similar(b)  # p*(1-p)
  tmp_n = similar(b)  # sample-space buffer
  tmp_v = similar(b)  # sample-space buffer

  # Numerically stable logistic sigmoid
  @inline function sigmoid(t)
    if t ≥ 0
      return 1 / (1 + exp(-t))
    else
      et = exp(t)
      return et / (1 + et)
    end
  end

  # Numerically stable softplus: log(1 + exp(t))
  @inline function softplus(t)
    t > 0 ? (t + log1p(exp(-t))) : log1p(exp(t))
  end

  # Objective: sum(softplus(A'x) - b .* (A'x))
  function obj(x)
    mul!(Ax, A', x)  # Ax = A' * x
    s = zero(T)
    @inbounds @simd for i in 1:n
      zi = Ax[i]
      s += softplus(zi) - b[i] * zi
    end
    return s
  end

  # Gradient: A * (sigmoid(A'x) - b)
  function grad!(g, x)
    mul!(Ax, A', x)

    @inbounds @simd for i in 1:n
      p[i] = sigmoid(Ax[i])
      tmp_n[i] = p[i] - b[i]
    end

    mul!(g, A, tmp_n)  # g = A * (p - b)
    return g
  end

  # Hessian-vector product: hv = A * ( (p .* (1-p)) .* (A'v) )
  function hprod!(hv, x, v; obj_weight = 1)
    mul!(Ax, A', x)

    @inbounds @simd for i in 1:n
      pi = sigmoid(Ax[i])
      w[i] = pi * (1 - pi)
    end

    mul!(tmp_v, A', v)   # tmp_v = A' * v
    @. tmp_v *= w        # tmp_v .= w .* tmp_v
    mul!(hv, A, tmp_v)   # hv = A * tmp_v

    if obj_weight != 1
      rmul!(hv, obj_weight)
    end
    return hv
  end

  x0 = fill!(similar(b, m), zero(T))

  # Construct an NLPModel with registered gradient + Hessian-vector product
  return ManualNLPModels.NLPModel(
    x0,
    obj;
    grad = grad!,
    hprod = hprod!,
    meta_args = Dict(:name => "Binomial"),
  )
end
