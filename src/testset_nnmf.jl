# Predefine a set of common problem instances.
export setup_nnmf_l0, setup_nnmf_l1

function setup_nnmf_l0(args...; kwargs...)
  model, nls_model, _, selected = nnmf_model(args...)
  λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 200
  h = ProximalOperators.NormL0(λ)
  return RegularizedNLPModel(model, h, selected),
  RegularizedNLSModel(nls_model, h, selected)
end

function setup_nnmf_l1(args...; kwargs...)
  model, nls_model, _, selected = nnmf_model(args...)
  λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 100_000
  h = ProximalOperators.NormL1(λ)
  return RegularizedNLPModel(model, h, selected),
  RegularizedNLSModel(nls_model, h, selected)
end
