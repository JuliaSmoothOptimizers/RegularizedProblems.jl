export AbstractRegularizedNLPModel, RegularizedNLPModel, RegularizedNLSModel

#! format: off
@deprecate FirstOrderModel(f, ∇f!, x; kwargs...) NLPModel(x, f; grad = ∇f!, meta_args = Dict(kwargs...))

@deprecate FirstOrderNLSModel(r!, jv!, jtv!, nequ, x; kwargs...) NLSModel(x, r!, nequ; jprod = jv!, jtprod = jtv!, kwargs...)
#! format: on

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

function NLPModels.hess_op(rnlp::AbstractRegularizedNLPModel, x::AbstractVector)
  return hess_op(rnlp.model, x)
end

function NLPModels.hess(rnlp::AbstractRegularizedNLPModel, x::AbstractVector; obj_weight=1.0)
  return hess(rnlp.model, x; obj_weight=obj_weight)
end

function NLPModels.hess(rnlp::AbstractRegularizedNLPModel, x::AbstractVector, y::AbstractVector; obj_weight=1.0)
  return hess(rnlp.model, x, y; obj_weight=obj_weight)
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
