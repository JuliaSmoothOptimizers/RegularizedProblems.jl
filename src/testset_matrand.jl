# Predefine a set of common problem instances.
export setup_random_completion_rank, setup_random_completion_nuclear
export setup_mit_completion_rank, setup_mit_completion_nuclear

function setup_random_completion_rank(args...; kwargs...)
  model, nls_model, _ = random_matrix_completion_model(; kwargs...)
  λ = 0.1
  h = Rank(λ)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_random_completion_nuclear(args...; kwargs...)
  model, nls_model, _ = random_matrix_completion_model(; kwargs...)
  λ = 0.1
  h = NuclearNorm(λ)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_mit_completion_rank(args...; kwargs...)
  model, nls_model, _ = MIT_matrix_completion_model()
  λ = 0.1
  h = Rank(λ)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_mit_completion_nuclear(args...; kwargs...)
  model, nls_model, _ = MIT_matrix_completion_model()
  λ = 0.1
  h = NuclearNorm(λ)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end
