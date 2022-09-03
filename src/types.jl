export FirstOrderModel, FirstOrderNLSModel

"""
    model = FirstOrderModel(f, ∇f!; name = "first-order model")

A simple subtype of `AbstractNLPModel` to represent a smooth objective.

## Arguments

* `f :: F <: Function`: a function such that `f(x)` returns the objective value at `x`;
* `∇f! :: G <: Function`: a function such that `∇f!(g, x)` stores the gradient of the
  objective at `x` in `g`;
* `x :: AbstractVector`: an initial guess.

## Keyword arguments

* `selected :: AbstractVector{<: Int}`: a list of variables to apply the regularizer to
    (default: all variables).

All other keyword arguments are passed through to the `NLPModelMeta` constructor.
"""
mutable struct FirstOrderModel{T, S, F, G, I, V} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters

  f::F
  ∇f!::G
  selected::V

  function FirstOrderModel{T, S, F, G, I, V}(
    f::F,
    ∇f!::G,
    x::S;
    selected::AbstractVector{I} = 1:length(x),
    kwargs...,
  ) where {T, S, F <: Function, G <: Function, I <: Integer, V <: AbstractVector{I}}
    nvar = length(x)
    slct = unique(selected)
    @assert all(1 .≤ slct .≤ nvar)
    meta = NLPModelMeta(nvar, x0 = x; kwargs...)
    return new{T, S, F, G, I, typeof(slct)}(meta, Counters(), f, ∇f!, slct)
  end
end

FirstOrderModel(
  f,
  ∇f!,
  x::S;
  selected::V = 1:length(x),
  kwargs...,
) where {S, I <: Integer, V <: AbstractVector{I}} =
  FirstOrderModel{eltype(S), S, typeof(f), typeof(∇f!), I, V}(
    f,
    ∇f!,
    x,
    selected = selected;
    kwargs...,
  )

function NLPModels.obj(nlp::FirstOrderModel, x::AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  return nlp.f(x)
end

function NLPModels.grad!(nlp::FirstOrderModel, x::AbstractVector, g::AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar x
  increment!(nlp, :neval_grad)
  nlp.∇f!(g, x)
  return g
end

"""
    model = FirstOrderNLSModel(r!, jv!, jtv!; name = "first-order NLS model")

A simple subtype of `AbstractNLSModel` to represent a nonlinear least-squares problem
with a smooth residual.

## Arguments

* `r! :: R <: Function`: a function such that `r!(y, x)` stores the residual at `x` in `y`;
* `jv! :: J <: Function`: a function such that `jv!(u, x, v)` stores the product between the residual Jacobian at `x` and the vector `v` in `u`;
* `jtv! :: Jt <: Function`: a function such that `jtv!(u, x, v)` stores the product between the transpose of the residual Jacobian at `x` and the vector `v` in `u`;
* `x :: AbstractVector`: an initial guess.

## Keyword arguments

* `selected :: AbstractVector{<: Int}`: a list of variables to apply the regularizer to
    (default: all variables).

All other keyword arguments are passed through to the `NLPModelMeta` constructor.
"""
mutable struct FirstOrderNLSModel{T, S, R, J, Jt, I, V} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters

  resid!::R
  jprod_resid!::J
  jtprod_resid!::Jt
  selected::V

  function FirstOrderNLSModel{T, S, R, J, Jt, I, V}(
    r::R,
    jv::J,
    jtv::Jt,
    nequ::Int,
    x::S;
    selected::AbstractVector{I} = 1:length(x),
    kwargs...,
  ) where {T, S, R <: Function, J <: Function, Jt <: Function, I <: Integer, V <: AbstractVector{I}}
    nvar = length(x)
    slct = unique(selected)
    @assert all(1 .≤ slct .≤ nvar)
    meta = NLPModelMeta(nvar, x0 = x; kwargs...)
    nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = x)
    return new{T, S, R, J, Jt, I, typeof(slct)}(meta, nls_meta, NLSCounters(), r, jv, jtv, slct)
  end
end

FirstOrderNLSModel(
  r,
  jv,
  jtv,
  nequ::Int,
  x::S;
  selected::V = 1:length(x),
  kwargs...,
) where {S, I <: Integer, V <: AbstractVector{I}} =
  FirstOrderNLSModel{eltype(S), S, typeof(r), typeof(jv), typeof(jtv), I, V}(
    r,
    jv,
    jtv,
    nequ,
    x,
    selected = selected;
    kwargs...,
  )

function NLPModels.residual!(nls::FirstOrderNLSModel, x::AbstractVector, Fx::AbstractVector)
  NLPModels.@lencheck nls.meta.nvar x
  NLPModels.@lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  nls.resid!(Fx, x)
  Fx
end

function NLPModels.jprod_residual!(
  nls::FirstOrderNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  NLPModels.@lencheck nls.meta.nvar x v
  NLPModels.@lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  nls.jprod_resid!(Jv, x, v)
  Jv
end

function NLPModels.jtprod_residual!(
  nls::FirstOrderNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  NLPModels.@lencheck nls.meta.nvar x Jtv
  NLPModels.@lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_jtprod_residual)
  nls.jtprod_resid!(Jtv, x, v)
  Jtv
end
