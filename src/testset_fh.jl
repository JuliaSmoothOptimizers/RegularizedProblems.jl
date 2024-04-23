# Predefine a set of common problem instances.
export setup_fh_l0, setup_fh_l1

function setup_fh_l0(; kwargs...)
  model, nls_model, _ = fh_model(; kwargs...)
  h = ProximalOperators.NormL0(1.0)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_fh_l1(; kwargs...)
  model, nls_model, _ = fh_model(; kwargs...)
  h = ProximalOperators.NormL1(10.0)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end
