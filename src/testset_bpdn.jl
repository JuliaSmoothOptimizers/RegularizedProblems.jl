# Predefine a set of common problem instances.
export setup_bpdn_l0, setup_bpdn_l1, setup_bpdn_B0

function setup_bpdn_l0(args...; kwargs...)
  model, nls_model, _ = bpdn_model(args...)
  λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
  h = NormL0(λ)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_bpdn_l1(args...; kwargs...)
  model, nls_model, _ = bpdn_model(args...)
  λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
  h = NormL1(λ)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_bpdn_B0(compound = 1, args...; kwargs...)
  model, nls_model, _ = bpdn_model(compound, args...)
  h = IndBallL0(10 * compound)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end
