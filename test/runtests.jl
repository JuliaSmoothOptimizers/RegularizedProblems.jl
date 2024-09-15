using LinearAlgebra, Test
using ADNLPModels, DifferentialEquations, NLPModels, MLDatasets, QuadraticModels
using Images, FFTW, Wavelets
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

  JF = jprod_residual(nls_model, x, x)
  @test JF' * F ≈ JtF' * x
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

@testset "Group-BPDN" begin
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
  model, nls_model, A = lrcomp_model()
  test_well_defined(model, nls_model, A)
  test_objectives(model, nls_model)
  @test model.meta.nvar == 10000
  @test all(0 .<= model.meta.x0 .<= 1)
  @test nls_model.nls_meta.nequ == 10000
  @test all(0 .<= nls_model.meta.x0 .<= 1)
end

@testset "mat_rand" begin
  model, nls_model, sol = random_matrix_completion_model()
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
  ENV["DATADEPS_ALWAYS_ACCEPT"] = true
  @test_throws ErrorException svm_train_model((1, 1))
  @test_throws MethodError svm_train_model((1, 2, 3))
  @test_throws ErrorException svm_train_model((10, -1))
  nlp_train, nls_train, sol = svm_train_model()
  @test typeof(nlp_train) <: FirstOrderModel
  @test typeof(nls_train) <: FirstOrderNLSModel
  @test typeof(sol) == Vector{Int64}
  test_objectives(nlp_train, nls_train)

  @test nlp_train.meta.nvar == 784
  @test nls_train.nls_meta.nequ == 13007
  @test all(nlp_train.meta.x0 .== 1)
  @test length(findall(x -> x .!= -1, sol)) == 6742
  @test length(findall(x -> x .!= 1, sol)) == 6265
end

@testset "SVM-Test" begin
  ENV["DATADEPS_ALWAYS_ACCEPT"] = true
  @test_throws ErrorException svm_test_model((1, 1))
  @test_throws MethodError svm_test_model((1, 2, 3))
  @test_throws ErrorException svm_test_model((10, -1))
  nlp_test, nls_test, sol = svm_test_model()
  @test typeof(nlp_test) <: FirstOrderModel
  @test typeof(nls_test) <: FirstOrderNLSModel
  @test typeof(sol) == Vector{Int64}
  test_objectives(nlp_test, nls_test)

  @test nlp_test.meta.nvar == 784
  @test nls_test.nls_meta.nequ == 2163
  @test all(nlp_test.meta.x0 .== 1)
  @test length(findall(x -> x .!= 1, sol)) == 1028
  @test length(findall(x -> x .!= -1, sol)) == 1135
end

@testset "NNMF" begin
  m, n, k = 100, 50, 10
  model, nls_model, sol, selected = nnmf_model(m, n, k)
  @test selected == (m * k + 1):((m + n) * k)
  test_well_defined(model, nls_model, sol)
  @test nls_model.nls_meta.nequ == m * n
  @test all(model.meta.lvar .== 0)
  @test all(model.meta.uvar .== Inf)
  @test all(nls_model.meta.lvar .== 0)
  @test all(nls_model.meta.uvar .== Inf)
  test_objectives(model, nls_model)
end

@testset "QP-rand" begin
  n, dens = 100, 0.1
  model = qp_rand_model(n; dens = dens, convex = false)
  @test all(-2.0 .≤ model.meta.lvar .≤ 0.0)
  @test all(0.0 .≤ model.meta.uvar .≤ 2.0)
  @test all(model.meta.x0 .== 0)

  model = qp_rand_model(n; dens = dens, convex = true)
  @test all(-2.0 .≤ model.meta.lvar .≤ 0.0)
  @test all(0.0 .≤ model.meta.uvar .≤ 2.0)
  @test all(model.meta.x0 .== 0)
end

include("rmodel_tests.jl")
@testset "denoising_model" begin
  n, m = 256, 256
  n_p, m_p = 260, 260
  kernel_size = 9
  model, sol = denoising_model((n, m), (n_p, m_p), kernel_size)
  @test typeof(model) <: FirstOrderModel
  @test typeof(sol) == typeof(model.meta.x0)
  @test model.meta.nvar == n * m
  x = model.meta.x0
  @test typeof(obj(model, x)) <: Float64
  @test typeof(grad(model, x)) <: Vector{Float64}
end
