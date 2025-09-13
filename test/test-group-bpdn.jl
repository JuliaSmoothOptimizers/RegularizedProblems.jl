model, nls_model, sol, g, ag, ind = group_lasso_model()
test_well_defined(model, nls_model, sol)
test_objectives(model, nls_model)
@test model.meta.nvar == 512
@test all(model.meta.x0 .== 0)
@test length(findall(x -> x .!= 0, sol)) / length(ag) == 32
@test nls_model.nls_meta.nequ == 200
@test g == 16
@test length(ag) == 5
@test size(ind) == (16, 32)
