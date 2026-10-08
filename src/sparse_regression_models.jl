export lasso_model, logistic_model

# Numerically stable scalar helpers for the logistic loss.
# log(1 + exp(t)) without overflow for large t.
_log1pexp(t::T) where {T <: Real} = t > 0 ? t + log1p(exp(-t)) : log1p(exp(t))
# σ(t) = 1 / (1 + exp(-t)) without overflow for large |t|.
function _sigmoid(t::T) where {T <: Real}
  if t ≥ 0
    return inv(one(T) + exp(-t))
  else
    e = exp(t)
    return e / (one(T) + e)
  end
end

"""
    model, nls_model = lasso_model(A, b; bounds = false, name = "Lasso")

Return an instance of an `NLPModel` and an instance of an `NLSModel` representing
the smooth part of the Lasso problem, i.e., the linear least-squares objective

    f(x) = ½ ‖Ax - b‖₂²,

for user-supplied data `A` (dense or sparse) and `b`.
The Lasso problem itself is obtained by adding the regularizer h(x) = λ ‖x‖₁, e.g.,
`RegularizedNLPModel(model, NormL1(λ))` with `NormL1` from ProximalOperators.jl.

The `NLPModel` implements `obj`, `grad!`, `objgrad!` and `hprod!` (Hessian AᵀA times a vector),
so it can be used directly with second-order solvers such as `R2N` (exact Hessian) or wrapped
in a quasi-Newton model from NLPModelsModifiers.jl.
The `NLSModel` implements the residual `Ax - b` and its Jacobian products.

The value of `Ax - b` at the last point where the objective or gradient was evaluated is cached,
so that evaluating `obj` and `grad` at the same point costs a single product with `A`.

## Arguments

* `A :: AbstractMatrix{T}`: the data matrix, with `T <: AbstractFloat`;
* `b :: AbstractVector`: the right-hand side, converted to `Vector{T}`.

## Keyword arguments

* `bounds :: Bool`: whether to add the nonnegativity constraints x ≥ 0 (default: `false`);
* `name :: AbstractString`: model name (default: `"Lasso"`, or `"Lasso-nonneg"` if `bounds == true`).

## Return Value

An instance of an `NLPModel` and an instance of an `NLSModel` that represent the same
least-squares problem. The initial guess is `x0 = 0`.
"""
function lasso_model(
  A::AbstractMatrix{T},
  b::AbstractVector;
  bounds::Bool = false,
  name::AbstractString = bounds ? "Lasso-nonneg" : "Lasso",
) where {T <: AbstractFloat}
  m, n = size(A)
  length(b) == m ||
    throw(DimensionMismatch("A has $m rows but b has $(length(b)) elements"))
  bv = convert(Vector{T}, b)  # new binding: reassigning `b` would box it in the closures

  r = zeros(T, m)        # residual Ax - b at the point xr
  xr = fill(T(NaN), n)   # point at which r is valid (NaN means "not computed yet")
  Av = zeros(T, m)       # work vector for Hessian-vector products

  function resid!(res, x)
    mul!(res, A, x)
    res .-= bv
    res
  end

  function update_residual!(x)
    if xr != x
      resid!(r, x)
      xr .= x
    end
    r
  end

  function obj(x)
    update_residual!(x)
    dot(r, r) / 2
  end

  function grad!(g, x)
    update_residual!(x)
    mul!(g, A', r)
    g
  end

  function objgrad!(g, x)
    update_residual!(x)
    mul!(g, A', r)
    dot(r, r) / 2, g
  end

  function hprod!(Hv, x, v; obj_weight = one(T))
    mul!(Av, A, v)
    mul!(Hv, A', Av)
    Hv .*= obj_weight
    Hv
  end

  jprod_resid!(Jv, x, v) = mul!(Jv, A, v)
  jtprod_resid!(Jtv, x, v) = mul!(Jtv, A', v)

  lvar = bounds ? zeros(T, n) : fill(T(-Inf), n)
  uvar = fill(T(Inf), n)

  nlp = NLPModel(
    zeros(T, n),
    obj;
    grad = grad!,
    objgrad = objgrad!,
    hprod = hprod!,
    lvar = lvar,
    uvar = uvar,
    meta_args = (name = name,),
  )
  nls = NLSModel(
    zeros(T, n),
    resid!,
    m;
    jprod = jprod_resid!,
    jtprod = jtprod_resid!,
    lvar = copy(lvar),
    uvar = copy(uvar),
    name = name * "-LS",
  )
  return nlp, nls
end

"""
    model = logistic_model(A, b; name = "Logistic")

Return an instance of an `NLPModel` representing the smooth part of the
ℓ₁-regularized logistic regression problem, i.e.,

    f(x) = ∑ᵢ log(1 + exp(-bᵢ aᵢᵀx)),

where aᵢᵀ is the i-th row of the data matrix `A` (dense or sparse) and `b` is the
vector of labels, normally in {-1, 1}.
Note that the loss is a sum, not an average, over the observations.
The regularized problem is obtained by adding, e.g., h(x) = λ ‖x‖₁.

The model implements `obj`, `grad!`, `objgrad!` and `hprod!` with the exact Hessian
∇²f(x) = Aᵀ D(x) A, where D(x) = diag(bᵢ² σ(tᵢ) (1 - σ(tᵢ))), tᵢ = -bᵢ aᵢᵀx and
σ is the sigmoid function.
All evaluations are numerically stable for large |aᵢᵀx|.
The product Ax at the last evaluated point and the Hessian weights D(x) are cached.

## Arguments

* `A :: AbstractMatrix{T}`: the data matrix, with `T <: AbstractFloat`;
* `b :: AbstractVector`: the labels, converted to `Vector{T}`.

## Keyword arguments

* `name :: AbstractString`: model name (default: `"Logistic"`).

## Return Value

An instance of an `NLPModel`. The initial guess is `x0 = 0`.
"""
function logistic_model(
  A::AbstractMatrix{T},
  b::AbstractVector;
  name::AbstractString = "Logistic",
) where {T <: AbstractFloat}
  m, n = size(A)
  length(b) == m ||
    throw(DimensionMismatch("A has $m rows but b has $(length(b)) elements"))
  bv = convert(Vector{T}, b)  # new binding: reassigning `b` would box it in the closures

  z = zeros(T, m)        # z = A x at the point xz
  xz = fill(T(NaN), n)
  u = zeros(T, m)        # work vector for gradients
  w = zeros(T, m)        # Hessian weights at the point xw
  xw = fill(T(NaN), n)
  Av = zeros(T, m)       # work vector for Hessian-vector products

  function update_Ax!(x)
    if xz != x
      mul!(z, A, x)
      xz .= x
    end
    z
  end

  function obj(x)
    update_Ax!(x)
    f = zero(T)
    @inbounds for i = 1:m
      f += _log1pexp(-bv[i] * z[i])
    end
    f
  end

  function grad!(g, x)
    update_Ax!(x)
    @inbounds for i = 1:m
      u[i] = -bv[i] * _sigmoid(-bv[i] * z[i])
    end
    mul!(g, A', u)
    g
  end

  function objgrad!(g, x)
    f = obj(x)
    grad!(g, x)
    f, g
  end

  function hprod!(Hv, x, v; obj_weight = one(T))
    if xw != x
      update_Ax!(x)
      @inbounds for i = 1:m
        s = _sigmoid(-bv[i] * z[i])
        w[i] = bv[i]^2 * s * (1 - s)
      end
      xw .= x
    end
    mul!(Av, A, v)
    Av .*= w
    mul!(Hv, A', Av)
    Hv .*= obj_weight
    Hv
  end

  return NLPModel(
    zeros(T, n),
    obj;
    grad = grad!,
    objgrad = objgrad!,
    hprod = hprod!,
    meta_args = (name = name,),
  )
end
