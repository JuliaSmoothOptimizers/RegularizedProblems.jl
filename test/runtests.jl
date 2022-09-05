using LinearAlgebra, Test
using ADNLPModels, DifferentialEquations, NLPModels
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
