# Predefine a set of common problem instances.
export setup_lrcomp_rank, setup_lrcomp_nuclear

function setup_lrcomp_rank(args...; kwargs...)
  model, nls_model, _ = lrcomp_model(; kwargs...)
  λ = 0.1
  h = ProximalOperators.Rank(λ)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_lrcomp_nuclear(args...; kwargs...)
  model, nls_model, _ = lrcomp_model(; kwargs...)
  λ = 0.1
  h = ProximalOperators.NuclearNorm(λ)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end
