# Predefine the regularized problems of Lopes, Santos and Silva (2019).
export setup_abcd_lasso_l1, setup_abcd_logistic_l1

"""
    rmodel, rnls_model, data = setup_abcd_lasso_l1(name; datadir = abcd_datadir(), λ = nothing)

Return the Lasso problem `name` of Lopes, Santos and Silva (2019),

    minimize ½‖Ax - b‖₂² + λ‖x‖₁  (subject to x ≥ 0 for `NN` problems),

as a `RegularizedNLPModel` and a `RegularizedNLSModel` with regularizer `NormL1(λ)`, together
with the data returned by `abcd_data`. By default, `λ = data.λ`, the value used in the paper.
The target objective value of the paper is `data.ftarget`.

This function is available after `using MAT, ProximalOperators`.
"""
function setup_abcd_lasso_l1(
  name::Union{AbstractString, Symbol};
  datadir::AbstractString = abcd_datadir(),
  λ::Union{Real, Nothing} = nothing,
)
  model, nls_model, data = abcd_lasso_model(name; datadir = datadir)
  h = ProximalOperators.NormL1(λ === nothing ? data.λ : λ)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h), data
end

"""
    rmodel, data = setup_abcd_logistic_l1(name; datadir = abcd_datadir(), λ = nothing)

Return the ℓ₁-regularized logistic regression problem `name` of Lopes, Santos and Silva (2019),

    minimize ∑ᵢ log(1 + exp(-bᵢ aᵢᵀx)) + λ‖x‖₁,

as a `RegularizedNLPModel` with regularizer `NormL1(λ)`, together with the data returned by
`abcd_data`. By default, `λ = data.λ`, the value used in the paper.
The target objective value of the paper is `data.ftarget`.

This function is available after `using MAT, ProximalOperators`.
"""
function setup_abcd_logistic_l1(
  name::Union{AbstractString, Symbol};
  datadir::AbstractString = abcd_datadir(),
  λ::Union{Real, Nothing} = nothing,
)
  model, data = abcd_logistic_model(name; datadir = datadir)
  h = ProximalOperators.NormL1(λ === nothing ? data.λ : λ)
  return RegularizedNLPModel(model, h), data
end
