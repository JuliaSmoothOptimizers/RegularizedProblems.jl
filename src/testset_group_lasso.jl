# Predefine a set of common problem instances.
export setup_group_lasso_l12

function setup_group_lasso_l12(args...; kwargs...)
  model, nls_model, ng, _, idx = group_lasso_model(; kwargs...)
  idx = [idx[i, :] for i = 1:ng]
  λ = 0.2 * ones(ng)
  h = GroupNormL2(λ, idx)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

