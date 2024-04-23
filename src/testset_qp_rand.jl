# Predefine a set of common problem instances.
export setup_qp_rand_l1

function setup_qp_rand_l1(args...; kwargs...)
  model, nls_model, _ = qp_rand_model(args...; kwargs...)
  λ = 0.1
  h = ProximalOperators.NormL1(λ)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end
