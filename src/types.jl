export FirstOrderNLSModel, AbstractRegularizedNLPModel, RegularizedNLPModel, RegularizedNLSModel

#! format: off
@deprecate FirstOrderModel(f, ∇f!, x; kwargs...) NLPModel(x, f; grad = ∇f!, meta_args = Dict(kwargs...))
#! format: on

"""
    model = FirstOrderNLSModel(r!, jv!, jtv!; name = "first-order NLS model")

A simple subtype of `AbstractNLSModel` to represent a nonlinear least-squares problem
with a smooth residual.

## Arguments

* `r! :: R <: Function`: a function such that `r!(y, x)` stores the residual at `x` in `y`;
* `jv! :: J <: Function`: a function such that `jv!(u, x, v)` stores the product between the residual Jacobian at `x` and the vector `v` in `u`;
* `jtv! :: Jt <: Function`: a function such that `jtv!(u, x, v)` stores the product between the transpose of the residual Jacobian at `x` and the vector `v` in `u`;
* `x :: AbstractVector`: an initial guess.

All keyword arguments are passed through to the `NLPModelMeta` constructor.
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
    kwargs...,
  ) where {T, S, R <: Function, J <: Function, Jt <: Function}
    nvar = length(x)
    meta = NLPModelMeta(nvar, x0 = x; kwargs...)
    nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = x)
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

abstract type AbstractRegularizedNLPModel{T, S} <: AbstractNLPModel{T, S} end

"""
    rmodel = RegularizedNLPModel(model, regularizer)
    rmodel = RegularizedNLSModel(model, regularizer)

An aggregate type to represent a regularized optimization model, .i.e.,
of the form

    minimize f(x) + h(x),

where f is smooth (and is usually assumed to have Lipschitz-continuous gradient),
and h is lower semi-continuous (and may have to be prox-bounded).

The regularized model is made of

- `model <: AbstractNLPModel`: the smooth part of the model, for example a `FirstOrderModel`
- `h`: the nonsmooth part of the model; typically a regularizer defined in `ProximalOperators.jl`
- `selected`: the subset of variables to which the regularizer h should be applied (default: all).

This aggregate type can be used to call solvers with a single object representing the
model, but is especially useful for use with SolverBenchmark.jl, which expects problems
to be defined by a single object.
"""
mutable struct RegularizedNLPModel{T, S, M <: AbstractNLPModel{T, S}, H, I} <:
               AbstractRegularizedNLPModel{T, S}
  model::M     # smooth  model
  h::H         # regularizer
  selected::I  # set of variables to which the regularizer should be applied
end

function RegularizedNLPModel(model::AbstractNLPModel{T, S}, h::H) where {T, S, H}
  selected = 1:get_nvar(model)
  RegularizedNLPModel{T, S, typeof(model), typeof(h), typeof(selected)}(model, h, selected)
end

mutable struct RegularizedNLSModel{T, S, M <: AbstractNLSModel{T, S}, H, I} <:
               AbstractRegularizedNLPModel{T, S}
  model::M     # smooth  model
  h::H         # regularizer
  selected::I  # set of variables to which the regularizer should be applied
end

function RegularizedNLSModel(model::AbstractNLSModel{T, S}, h::H) where {T, S, H}
  selected = 1:get_nvar(model)
  RegularizedNLSModel{T, S, typeof(model), typeof(h), typeof(selected)}(model, h, selected)
end

function NLPModels.obj(rnlp::AbstractRegularizedNLPModel, x::AbstractVector)
  # The size check on x will be performed when evaluating the smooth model.
  # We intentionally do not increment an objective evaluation counter here
  # because the relevant counters are inside the smooth term.
  obj(rnlp.model, x) + rnlp.h(x)
end

# Forward meta getters so they grab info from the smooth model
for field ∈ fieldnames(NLPModels.NLPModelMeta)
  meth = Symbol("get_", field)
  if field == :name
    @eval NLPModels.$meth(rnlp::RegularizedNLPModel) =
      NLPModels.$meth(rnlp.model) * "/" * string(typeof(rnlp.h).name.wrapper)
    @eval NLPModels.$meth(rnls::RegularizedNLSModel) =
      NLPModels.$meth(rnls.model) * "/" * string(typeof(rnls.h).name.wrapper)
  else
    @eval NLPModels.$meth(rnlp::RegularizedNLPModel) = NLPModels.$meth(rnlp.model)
  end
end

for field in fieldnames(NLPModels.NLSMeta)
  meth = Symbol("get_", field)
  @eval NLPModels.$meth(rnls::RegularizedNLSModel) = NLPModels.$meth(rnls.model)
end

# Forward counter getters so they grab info from the smooth model
for model_type ∈ (RegularizedNLPModel, RegularizedNLSModel)
  for counter in fieldnames(Counters)
    @eval NLPModels.$counter(rnlp::$model_type) = NLPModels.$counter(rnlp.model)
  end
end

for counter in fieldnames(NLSCounters)
  counter == :counters && continue
  @eval NLPModels.$counter(rnls::RegularizedNLSModel) = NLPModels.$counter(rnls.model)
end

# simple show method for now
function Base.show(io::IO, rnlp::AbstractRegularizedNLPModel)
  print(io, "Smooth model: ")
  show(io, rnlp.model)
  print(io, "\nRegularizer: ")
  show(io, rnlp.h)
  print(io, "\n\nSelected variables: ")
  show(io, rnlp.selected)
end
