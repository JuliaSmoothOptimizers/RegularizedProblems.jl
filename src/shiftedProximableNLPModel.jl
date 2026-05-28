export AbstractShiftedProximableNLPModel, ShiftedProximableQuadraticNLPModel
export set_sigma!, get_sigma

abstract type AbstractShiftedProximableNLPModel{T, V} <: AbstractRegularizedNLPModel{T, V} end

"""
    subproblem = ShiftedProximableQuadraticNLPModel(reg_nlp, x; kwargs...)

Given a regularized NLP model `reg_nlp` representing the problem

    minimize f(x) + h(x)  subject to l ≤ x ≤ u,
  
construct a shifted quadratic model around `x`:

    minimize  φ(s; x) + ½ σ ‖s‖² + ψ(s; x) + χ(s; [l - x, u - x] ∩ ΔB )

where φ(s ; x) = f(x) + ∇f(x)ᵀs + ½ sᵀBs is a quadratic approximation of f about x,
ψ(s; x) is either h(x + s) or an approximation of h(x + s),
‖⋅‖ is the ℓ₂ norm and σ > 0 is the regularization parameter,
χ(s; [l - x, u - x] ∩ ΔB ) is an indicator function over the intersection of the box [l - x, u - x] and the ball ΔB of radius Δ in the infinity norm.
In the case where there are no bounds (i.e., l = -∞ and u = +∞), the ball can be defined in any norm.

The ShiftedProximableQuadraticNLPModel is made of the following components:

- `model <: AbstractNLPModel`: represents φ + ½ σ ‖s‖², the quadratic approximation of the smooth part of the objective function (up to the constant term f(x));
- `h <: ShiftedProximableFunction`: represents ψ, the shifted version of the nonsmooth part of the model;
- `selected`: the subset of variables to which the regularizer h should be applied (default: all).
- `χ`: the indicator function of the intersection between the box defined by the bounds and the ball defined by the trust region radius.
- `Δ`: the trust region radius.
- `parent`: the original regularized NLP model from which the subproblem was derived.

# Arguments
- `reg_nlp::AbstractRegularizedNLPModel{T, V}`: the regularized NLP model for which the subproblem is being constructed.
- `x::V`: the point around which the quadratic model is constructed.

# Keyword Arguments
- `∇f::VNG = nothing`: the gradient of the smooth part of the objective function at `x`. If not provided, it will be computed.
- `indicator_type::Symbol = :none`: the type of indicator function to use for χ. It can be one of `:box`, `:ball`, or `:none`. 
  - If `:box`, χ is the indicator function over a box.
  - If `:ball`, χ is the indicator function over a ball.
  - If `:none`, χ is not included in the model.
- `tr_norm = NormLinf(T(1))`: the norm to use for the trust region when `indicator_type` is `:ball`. When `indicator_type` is `:box` or `:none`, this argument is ignored.
- `Δ::T = T(Inf)`: the radius of the trust region. When `indicator_type` is `:none`, this argument is ignored.

The matrix B is constructed as a `LinearOperator` and is the returned value of `hess_op(reg_nlp, x)` (see https://jso.dev/NLPModels.jl/stable/reference/#NLPModels.hess_op).
φ is constructed as a `QuadraticModel`, (see https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl).
When there are bounds, the shifted bounds l-x and u-x are stored in the metadata of the quadratic model φ.

# NLPModels Interface
The `ShiftedProximableQuadraticNLPModel` implements the `obj` function from the `NLPModels` interface, which evaluates the objective function φ(s; x) + ½ σ ‖s‖² + ψ(s; x) at a given point s.
The `obj` function has two additional optional keyword arguments: `skip_sigma` (default: false) and `cauchy` (default: false). 
If `skip_sigma` is true, the term ½ σ ‖s‖² is not included in the evaluation of the objective.
If `cauchy` is true, the term `B` is not included in the evaluation of the objective.
"""
mutable struct ShiftedProximableQuadraticNLPModel{T, V, M <: AbstractNLPModel{T, V}, H <: ShiftedProximalOperators.ShiftedProximableFunction, I, X, P <: AbstractRegularizedNLPModel{T, V}} <:
       AbstractShiftedProximableNLPModel{T, V}
  model::M
  h::H
  selected::I
  χ::X
  Δ::T
  parent::P
end

function ShiftedProximableQuadraticNLPModel(
  reg_nlp::AbstractRegularizedNLPModel{T, V}, 
  x::V;
  ∇f::VN = nothing,
  indicator_type::Symbol = :none,
  tr_norm = ProximalOperators.NormLinf(T(1)),
  Δ::T = T(Inf),
) where {T, V, VN <: Union{V, Nothing}}
  @assert indicator_type ∈ (:box, :ball, :none) "indicator_type must be one of :box, :ball, or :none"
  
  nlp, h, selected = reg_nlp.model, reg_nlp.h, reg_nlp.selected

  # φ(s) + ½ σ ‖s‖²
  B = hess_op(reg_nlp, x)
  isnothing(∇f) && (∇f = grad(nlp, x))
  φ = QuadraticModel(∇f, B, x0 = x, regularize = true)

  # χ(s)
  l_bound_m_x, u_bound_m_x = φ.meta.lvar, φ.meta.uvar
  indicator_type = has_bounds(nlp) ? :box : indicator_type
  χ = nothing
  if indicator_type == :box
    χ = BoxIndicatorFunction(zero(l_bound_m_x), zero(u_bound_m_x))
    @. χ.l = max(nlp.meta.lvar - x, -Δ)
    @. χ.u = min(nlp.meta.uvar - x, Δ)
  elseif indicator_type == :ball
    χ = BallIndicatorFunction(Δ, tr_norm)
  end

  # ψ(s) + χ(s)
  # FIXME: the indicator function logic can (and should) be simplified in `ShiftedProximalOperators.jl`...
  # FIXME: `shifted` call ignores the `selected` argument when there are no bounds!
  ψ = indicator_type == :box ?
    ShiftedProximalOperators.shifted(h, x, χ.l, χ.u, selected) :
    indicator_type == :ball ?
      ShiftedProximalOperators.shifted(h, x, χ.Δ, χ.norm) : 
      ShiftedProximalOperators.shifted(h, x)
      
  ShiftedProximableQuadraticNLPModel(φ, ψ, selected, χ, Δ, reg_nlp)
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
  φ, ψ, χ = reg_nlp.model, reg_nlp.h, reg_nlp.χ

  if isa(χ, BoxIndicatorFunction)
    @. φ.meta.lvar = nlp.meta.lvar - x
    @. φ.meta.uvar = nlp.meta.uvar - x
    ShiftedProximalOperators.set_radius!(reg_nlp, reg_nlp.Δ)
  end

  ShiftedProximalOperators.shift!(ψ, x)

  g = φ.data.c
  compute_grad && grad!(nlp, x, g)

  φ.data.H = hess_op(nlp, x)
end

function NLPModels.obj(reg_nlp::AbstractShiftedProximableNLPModel, s::AbstractVector; skip_sigma::Bool = false, cauchy::Bool = false)
  φ, ψ = reg_nlp.model, reg_nlp.h

  σ_temp = get_sigma(reg_nlp)
  σ_c = skip_sigma ? zero(σ_temp) : σ_temp
  set_sigma!(reg_nlp, σ_c)

  φs = cauchy ? dot(φ.data.c, s) + σ_c * dot(s, s)/2 : obj(φ, s)
  ψs = ψ(s)

  set_sigma!(reg_nlp, σ_temp) # restore original σ

  return φs + ψs
end

function get_sigma(reg_nlp::ShiftedProximableQuadraticNLPModel{T, V}) where {T, V}
  φ = reg_nlp.model
  return φ.data.σ
end

function set_sigma!(
  reg_nlp::ShiftedProximableQuadraticNLPModel{T, V},
  σ::T
) where {T, V}
  φ = reg_nlp.model
  φ.data.σ = σ
end

function ShiftedProximalOperators.set_radius!(
  reg_nlp::ShiftedProximableQuadraticNLPModel{T, V},
  Δ::T
) where {T, V}
  φ, ψ, χ = reg_nlp.model, reg_nlp.h, reg_nlp.χ

  # Update Radius
  reg_nlp.Δ = Δ

  # Update Bounds if necessary
  if isa(χ, BallIndicatorFunction)
    ShiftedProximalOperators.set_radius!(ψ, Δ)
    χ.Δ = Δ
  elseif isa(χ, BoxIndicatorFunction)
    @. χ.l = max(φ.meta.lvar, -Δ)
    @. χ.u = min(φ.meta.uvar, Δ)
  end

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

# Indicator functions logic
abstract type AbstractIndicatorFunction end

mutable struct BoxIndicatorFunction{V} <: AbstractIndicatorFunction
  l::V
  u::V
end

mutable struct BallIndicatorFunction{T, N} <: AbstractIndicatorFunction
  Δ::T
  norm::N
end

