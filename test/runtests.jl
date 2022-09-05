using LinearAlgebra, Test
using ADNLPModels, DifferentialEquations, NLPModels, MLDatasets
using RegularizedProblems

function test_well_defined(model, nls_model, sol)
  @test typeof(model) <: FirstOrderModel
  @test typeof(sol) == typeof(model.meta.x0)
  @test typeof(nls_model) <: FirstOrderNLSModel
  @test model.meta.nvar == nls_model.meta.nvar
  @test all(model.meta.x0 .== nls_model.meta.x0)
end

function test_objectives(model, nls_model, x = model.meta.x0)
  f = obj(model, x)
  F = residual(nls_model, x)
  @test f ≈ dot(F, F) / 2

  g = grad(model, x)
  JtF = jtprod_residual(nls_model, x, F)
  @test all(g .≈ JtF)
end

@testset "BPDN" begin
  model, nls_model, sol = bpdn_model()
  test_well_defined(model, nls_model, sol)
  test_objectives(model, nls_model)
  @test model.meta.nvar == 512
  @test all(model.meta.x0 .== 0)
  @test length(findall(x -> x .!= 0, sol)) == 10
  @test nls_model.nls_meta.nequ == 200
end

@testset "BPDN with bounds" begin
  model, nls_model, sol = bpdn_model(; bounds = true)
  test_well_defined(model, nls_model, sol)
  test_objectives(model, nls_model)
  @test all(sol .≥ 0)
  @test has_bounds(model)
  @test all(model.meta.lvar .== 0)
  @test all(model.meta.uvar .== Inf)
  @test has_bounds(nls_model)
  @test all(nls_model.meta.lvar .== 0)
  @test all(nls_model.meta.uvar .== Inf)
end

@testset "FH" begin
  model, nls_model, sol = fh_model()
  test_objectives(model, nls_model)
  @test typeof(model) <: ADNLPModel
  @test typeof(sol) == typeof(model.meta.x0)
  @test model.meta.nvar == 5
  @test all(model.meta.x0 .== 1)
  @test length(findall(x -> x .!= 0, sol)) == 2
  @test typeof(nls_model) <: ADNLSModel
  @test nls_model.meta.nvar == 5
  @test nls_model.nls_meta.nequ == 202
  @test all(nls_model.meta.x0 .== 1)
end

@testset "low-rank completion" begin
  model, nls_model, A = lrcomp_model(100, 100)
  test_well_defined(model, nls_model, A)
  test_objectives(model, nls_model)
  @test model.meta.nvar == 10000
  @test all(0 .<= model.meta.x0 .<= 1)
  @test nls_model.nls_meta.nequ == 10000
  @test all(0 .<= nls_model.meta.x0 .<= 1)
end

@testset "mat_rand" begin
  model, nls_model, sol = random_matrix_completion_model(100, 100, 5, 0.8, 0.0001, 0.01, 0.2)
  test_well_defined(model, nls_model, sol)
  test_objectives(model, nls_model)
  @test model.meta.nvar == 10000
  @test all(0 .<= model.meta.x0 .<= 1)
  @test nls_model.nls_meta.nequ == 10000
  @test all(0 .<= nls_model.meta.x0 .<= 1)
end

@testset "MIT" begin
  model, nls_model, sol = MIT_matrix_completion_model()
  test_well_defined(model, nls_model, sol)
  test_objectives(model, nls_model)
  @test model.meta.nvar == 256 * 256
  @test all(0 .<= model.meta.x0 .<= 1)
  @test nls_model.nls_meta.nequ == 256 * 256
  @test all(0 .<= nls_model.meta.x0 .<= 1)
end

@testset "SVM-Train" begin
  nlp_train,nls_train,sol = svm_train_model()
  @test typeof(nlp_train) <: FirstOrderModel
  @test typeof(nls_train) <: FirstOrderNLSModel
  @test typeof(sol) == Vector{Int64}


  x = nlp_train.meta.x0
  f = obj(nlp_train, x)
  F = residual(nls_train, x)
  @test f ≈ dot(F, F) / 2

  g = grad(nlp_train, x)
  JtF = jtprod_residual(nls_train, x, F)
  @test all(g .≈ JtF)
  ### below is TBD
  # @test model.meta.nvar == 5
  # @test model.nls_meta.nequ == 202
  # @test all(model.meta.x0 .== 1)
  # @test length(findall(x -> x .!= 0, sol)) == 2
end

@testset "SVM-Test" begin
  nlp_test,nls_test,sol = svm_test_model()
  @test typeof(nlp_test) <: FirstOrderModel
  @test typeof(nls_test) <: FirstOrderNLSModel
  @test typeof(sol) == Vector{Int64}

  x = nlp_test.meta.x0
  f = obj(nlp_test, x)
  F = residual(nls_test, x)
  @test f ≈ dot(F, F) / 2

  g = grad(nlp_test, x)
  JtF = jtprod_residual(nls_test, x, F)
  @test all(g .≈ JtF)
  ### Below is TBD
  # @test model.meta.nvar == 5
  # @test model.nls_meta.nequ == 202
  # @test all(model.meta.x0 .== 1)
  # @test length(findall(x -> x .!= 0, sol)) == 2
end

##### Obsolete since svm_train/test_model() does not yield resid() anymore
# function comp_derivs()
#   nlp_train, nlp_test, sol_train = svm_train_model() #
# nls_train = ADNLSModel(resid_train, ones(size(nls_train.meta.x0)),size(sol_train,1) + size(nls_train.meta.x0,1))
# f = ReverseADNLSModel(resid!, size(sol_train,1), ones(size(model_train.meta.x0)), name = "Dominique")
#   fad = ReverseADNLSModel(resid!, size(sol_train,1), ones(size(model_train.meta.x0)), name = "Dominique")

#   xk = 10*randn(size(fk.meta.x0));
#   v = 10*randn(size(xk));
#   Jvac = zeros(size(sol_train));
#   Jvac_ad = similar(Jvac);
#   @show @benchmark jprod_residual!($fk, $xk, $v, $Jvac)
#   @show @benchmark jprod_residual!($fad, $xk, $v, $Jvac_ad)

#   # @show norm(Jvac - Jvac_ad)

#   v = 10*randn(size(sol_train));
#   Jtvac = zero(xk);
#   Jtvac_ad = zero(xk);

#   @show @benchmark jtprod_residual!($fk, $xk, $v, $Jtvac)
#   @show @benchmark jtprod_residual!($fad, $xk, $v, $Jtvac_ad)

#   @show norm(Jtvac - Jtvac_ad)

# end