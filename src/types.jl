export FirstOrderModel, FirstOrderNLSModel

"""
    model = FirstOrderModel(f, ∇f!; name = "first-order model")

A simple subtype of `AbstractNLPModel` to represent a smooth objective.

## Arguments

* `f :: F <: Function`: a function such that `f(x)` returns the objective value at `x`;
* `∇f! :: G <: Function`: a function such that `∇f!(g, x)` stores the gradient of the
  objective at `x` in `g`.
"""
mutable struct FirstOrderModel{T, S, F, G} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters

  f::F
  ∇f!::G

  function FirstOrderModel{T, S, F, G}(
    f::F,
    ∇f!::G,
    x::S;
    name::AbstractString = "first-order model",
    uvar::S = nothing,
    lvar::S = nothing,
  ) where {T, S, F <: Function, G <: Function}
    if uvar != nothing & lvar != nothing
      meta = NLPModelMeta(length(x), x0 = x, name = name, lvar = lvar, uvar = uvar)
    else 
      meta = NLPModelMeta(length(x), x0 = x, name = name)
    end 
    return new{T, S, F, G}(meta, Counters(), f, ∇f!)
  end
end

FirstOrderModel(f, ∇f!, x::S; kwargs...) where {S} =
  FirstOrderModel{eltype(S), S, typeof(f), typeof(∇f!)}(f, ∇f!, x; kwargs...)

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
* `jtv! :: Jt <: Function`: a function such that `jtv!(u, x, v)` stores the product between the transpose of the residual Jacobian at `x` and the vector `v` in `u`.
"""
mutable struct FirstOrderNLSModel{T, S, R, J, Jt} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters

  resid!::R
  jprod_resid!::J
  jtprod_resid!::Jt

  function FirstOrderNLSModel{T, S, R, J, Jt}(
    r::R,
    jv::J,
    jtv::Jt,
    nequ::Int,
    x::S;
    name::AbstractString = "first-order NLS model",
  ) where {T, S, R <: Function, J <: Function, Jt <: Function}
    meta = NLPModelMeta(length(x), x0 = x, name = name)
    nls_meta = NLSMeta{T, S}(nequ, length(x), x0 = x)
    return new{T, S, R, J, Jt}(meta, nls_meta, NLSCounters(), r, jv, jtv)
  end
end

FirstOrderNLSModel(r, jv, jtv, nequ::Int, x::S; kwargs...) where {S} =
  FirstOrderNLSModel{eltype(S), S, typeof(r), typeof(jv), typeof(jtv)}(
    r,
    jv,
    jtv,
    nequ,
    x;
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
