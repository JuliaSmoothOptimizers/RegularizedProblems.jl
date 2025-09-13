ENV["DATADEPS_ALWAYS_ACCEPT"] = true
@test_throws ErrorException svm_test_model((1, 1))
@test_throws MethodError svm_test_model((1, 2, 3))
@test_throws ErrorException svm_test_model((10, -1))
nlp_test, nls_test, sol = svm_test_model()
@test typeof(nlp_test) <: NLPModel
@test typeof(nls_test) <: NLSModel
@test typeof(sol) == Vector{Int64}
test_objectives(nlp_test, nls_test)

@test nlp_test.meta.nvar == 784
@test nls_test.nls_meta.nequ == 2163
@test all(nlp_test.meta.x0 .== 1)
@test length(findall(x -> x .!= 1, sol)) == 1028
@test length(findall(x -> x .!= -1, sol)) == 1135
