ENV["DATADEPS_ALWAYS_ACCEPT"] = true
@test_throws ErrorException svm_train_model((1, 1))
@test_throws MethodError svm_train_model((1, 2, 3))
@test_throws ErrorException svm_train_model((10, -1))
nlp_train, nls_train, sol = svm_train_model()
@test typeof(nlp_train) <: NLPModel
@test typeof(nls_train) <: NLSModel
@test typeof(sol) == Vector{Int64}
test_objectives(nlp_train, nls_train)

@test nlp_train.meta.nvar == 784
@test nls_train.nls_meta.nequ == 13007
@test all(nlp_train.meta.x0 .== 1)
@test length(findall(x -> x .!= -1, sol)) == 6742
@test length(findall(x -> x .!= 1, sol)) == 6265
