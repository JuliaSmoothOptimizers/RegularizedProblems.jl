# Predefine a set of common problem instances.
export setup_bpdn_l0, setup_bpdn_l1, setup_bpdn_B0

function setup_bpdn_l0(args...; kwargs...)
  model, nls_model, _ = bpdn_model(args...; kwargs...)
  y = similar(model.meta.x0)
  grad!(model, y, zeros(model.meta.nvar))
  λ = norm(y) / 10
  h = ProximalOperators.NormL0(λ)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_bpdn_l1(args...; kwargs...)
  model, nls_model, _ = bpdn_model(args...; kwargs...)
  y = similar(model.meta.x0)
  grad!(model, y, zeros(model.meta.nvar))
  λ = norm(y) / 10
  h = ProximalOperators.NormL1(λ)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_bpdn_B0(compound = 1, args...; kwargs...)
  model, nls_model, _ = bpdn_model(compound, args...; kwargs...)
  h = ProximalOperators.IndBallL0(10 * compound)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end
