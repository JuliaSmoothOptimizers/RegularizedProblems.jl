m, n, k = 100, 50, 10
model, nls_model, sol, selected = nnmf_model(m, n, k)
@test selected == (m*k+1):((m+n)*k)
test_well_defined(model, nls_model, sol)
@test nls_model.nls_meta.nequ == m * n
@test all(model.meta.lvar .== 0)
@test all(model.meta.uvar .== Inf)
@test all(nls_model.meta.lvar .== 0)
@test all(nls_model.meta.uvar .== Inf)
test_objectives(model, nls_model)
