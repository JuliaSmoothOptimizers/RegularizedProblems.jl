model, nls_model, sol = bpdn_model()
test_well_defined(model, nls_model, sol)
test_objectives(model, nls_model)
@test model.meta.nvar == 512
@test all(model.meta.x0 .== 0)
@test length(findall(x -> x .!= 0, sol)) == 10
@test nls_model.nls_meta.nequ == 200
