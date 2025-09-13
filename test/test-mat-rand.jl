model, nls_model, sol = random_matrix_completion_model()
test_well_defined(model, nls_model, sol)
test_objectives(model, nls_model)
@test model.meta.nvar == 10000
@test all(0 .<= model.meta.x0 .<= 1)
@test nls_model.nls_meta.nequ == 10000
@test all(0 .<= nls_model.meta.x0 .<= 1)
