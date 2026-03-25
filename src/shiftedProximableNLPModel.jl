export AbstractShiftedProximableNLPModel, ShiftedProximableQuadraticNLPModel
export update_sigma!, get_sigma

abstract type AbstractShiftedProximableNLPModel{T, V} <: AbstractRegularizedNLPModel{T, V} end

"""
    subproblem = ShiftedProximableQuadraticNLPModel(reg_nlp, x; kwargs...)

Given a regularized NLP model `reg_nlp` representing the problem

    minimize f(x) + h(x),
  
construct a shifted quadratic model around `x`:

    minimize  φ(s; x) + ½ σ ‖s‖² + ψ(s; x),

where φ(s ; x) = f(x) + ∇f(x)ᵀs + ½ sᵀBs is a quadratic approximation of f about x,
ψ(s; x) is either h(x + s) or an approximation of h(x + s), ‖⋅‖ is the ℓ₂ norm and σ > 0 is the regularization parameter.

The ShiftedProximableQuadraticNLPModel is made of the following components:

- `model <: AbstractNLPModel`: represents φ + ½ σ ‖s‖², the quadratic approximation of the smooth part of the objective function (up to the constant term f(x));
- `h <: ShiftedProximableFunction`: represents ψ, the shifted version of the nonsmooth part of the model;
- `selected`: the subset of variables to which the regularizer h should be applied (default: all).
- `parent`: the original regularized NLP model from which the subproblem was derived.

# Arguments
- `reg_nlp::AbstractRegularizedNLPModel{T, V}`: the regularized NLP model for which the subproblem is being constructed.
- `x::V`: the point around which the quadratic model is constructed.

# Keyword Arguments
- `l_bound_m_x::VN = nothing`: the vector of lower bounds minus `x` (i.e., l - x), required if the original NLP model has bounds.
- `u_bound_m_x::VN = nothing`: the vector of upper bounds minus `x` (i.e., u - x), required if the original NLP model has bounds.
- `∇f::VNG = nothing`: the gradient of the smooth part of the objective function at `x`. If not provided, it will be computed.

The matrix B is constructed as a `LinearOperator` and is the returned value of `hess_op(reg_nlp, x)` (see https://jso.dev/NLPModels.jl/stable/reference/#NLPModels.hess_op).
φ is constructed as a `QuadraticModel`, (see https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl).
"""
mutable struct ShiftedProximableQuadraticNLPModel{T, V, M <: AbstractNLPModel{T, V}, H <: ShiftedProximalOperators.ShiftedProximableFunction, I, P <: AbstractRegularizedNLPModel{T, V}} <:
       AbstractShiftedProximableNLPModel{T, V}
  model::M
  h::H
  selected::I
  parent::P
end

function ShiftedProximableQuadraticNLPModel(
  reg_nlp::AbstractRegularizedNLPModel{T, V}, 
  x::V;
  l_bound_m_x::VN = nothing,
  u_bound_m_x::VN = nothing,
  ∇f::VNG = nothing,
) where {T, V, VN <: Union{V, Nothing}, VNG <: Union{V, Nothing}}
  nlp, h, selected = reg_nlp.model, reg_nlp.h, reg_nlp.selected

  if (has_bounds(nlp) && isnothing(l_bound_m_x) && isnothing(u_bound_m_x)) 
    l_bound_m_x, u_bound_m_x = copy(nlp.meta.lvar), copy(nlp.meta.uvar)
    l_bound_m_x .-= x
    u_bound_m_x .-= x
  end

  # FIXME: `shifted` call ignores the `selected` argument when there are no bounds!
  ψ = has_bounds(nlp) ? ShiftedProximalOperators.shifted(h, x, l_bound_m_x, u_bound_m_x, selected) : ShiftedProximalOperators.shifted(h, x)

  B = hess_op(reg_nlp, x)
  isnothing(∇f) && (∇f = grad(nlp, x))
  φ = QuadraticModel(∇f, B, x0 = x, regularize = true)

  ShiftedProximableQuadraticNLPModel(φ, ψ, selected, reg_nlp)
end

"""
    shift!(reg_nlp::ShiftedProximableQuadraticNLPModel, x; compute_grad = true)

Update the shifted quadratic model `reg_nlp` at the point `x`. 
i.e. given the shifted quadratic model around `y`:

    minimize  φ(s; y) + ½ σ ‖s‖² + ψ(s; y),

update it to be around `x`:

    minimize  φ(s; x) + ½ σ ‖s‖² + ψ(s; x).

# Arguments
- `reg_nlp::ShiftedProximableQuadraticNLPModel`: the shifted quadratic model to be updated.
- `x::V`: the point around which the shifted quadratic model should be updated.

# Keyword Arguments
- `compute_grad::Bool = true`: whether the gradient of the smooth part of the model should be updated.
"""
function ShiftedProximalOperators.shift!(
  reg_nlp::ShiftedProximableQuadraticNLPModel{T, V},
  x::V;
  compute_grad::Bool = true
) where{T, V}
  nlp, h = reg_nlp.parent.model, reg_nlp.parent.h
  φ, ψ = reg_nlp.model, reg_nlp.h

  if has_bounds(nlp)
    @. ψ.l = nlp.meta.lvar - x
    @. ψ.u = nlp.meta.uvar - x
  end
  ShiftedProximalOperators.shift!(ψ, x)

  g = φ.data.c
  compute_grad && grad!(nlp, x, g)

  if NLPModels.has_hess(nlp)
    φ.data.H = NLPModels.hess_op(nlp, x)
  end
end

function NLPModels.obj(reg_nlp::AbstractShiftedProximableNLPModel, s::AbstractVector; skip_sigma::Bool = false, cauchy::Bool = false)
  φ, ψ = reg_nlp.model, reg_nlp.h

  σ_temp = get_sigma(reg_nlp)
  σ_c = skip_sigma ? zero(σ_temp) : σ_temp
  update_sigma!(reg_nlp, σ_c)

  φs = cauchy ? dot(φ.data.c, s) + σ_c * dot(s, s)/2 : obj(φ, s)
  ψs = ψ(s)

  update_sigma!(reg_nlp, σ_temp) # restore original σ

  return φs + ψs
end

function get_sigma(reg_nlp::ShiftedProximableQuadraticNLPModel{T, V}) where {T, V}
  φ = reg_nlp.model
  return φ.data.σ
end

function update_sigma!(
  reg_nlp::ShiftedProximableQuadraticNLPModel{T, V},
  σ::T
) where {T, V}
  φ = reg_nlp.model
  φ.data.σ = σ
end

# Forward meta getters so they grab info from the smooth model
for field ∈ fieldnames(NLPModels.NLPModelMeta)
  meth = Symbol("get_", field)
  if field == :name
    @eval NLPModels.$meth(rnlp::ShiftedProximableQuadraticNLPModel) =
      NLPModels.$meth(rnlp.model) * "/" * string(typeof(rnlp.h).name.wrapper)
  else
    @eval NLPModels.$meth(rnlp::ShiftedProximableQuadraticNLPModel) = NLPModels.$meth(rnlp.model)
  end
end

# Forward counter getters so they grab info from the smooth model
for model_type ∈ (ShiftedProximableQuadraticNLPModel,)
  for counter in fieldnames(Counters)
    @eval NLPModels.$counter(rnlp::$model_type) = NLPModels.$counter(rnlp.model)
  end
end

