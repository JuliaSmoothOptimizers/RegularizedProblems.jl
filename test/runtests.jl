using Test
using ADNLPModels, DifferentialEquations
using RegularizedProblems

@testset "BPDN" begin
  model, sol = bpdn_model()
  @test typeof(model) <: FirstOrderModel
  @test typeof(sol) == Vector{Float64}
  @test model.meta.nvar == 512
  @test all(model.meta.x0 .== 0)
  @test length(findall(x -> x .!= 0, sol)) == 10
end

@testset "BPDN-LS" begin
  model, sol = bpdn_nls_model()
  @test typeof(model) <: FirstOrderNLSModel
  @test typeof(sol) == Vector{Float64}
  @test model.meta.nvar == 512
  @test model.nls_meta.nequ == 200
  @test all(model.meta.x0 .== 0)
  @test length(findall(x -> x .!= 0, sol)) == 10
end

@testset "FH" begin
  model, sol = fh_model()
  @test typeof(model) <: ADNLPModel
  @test typeof(sol) == Vector{Float64}
  @test model.meta.nvar == 5
  @test all(model.meta.x0 .== 1)
  @test length(findall(x -> x .!= 0, sol)) == 2
end

@testset "FH-NLS" begin
  model, sol = fh_nls_model()
  @test typeof(model) <: ADNLSModel
  @test typeof(sol) == Vector{Float64}
  @test model.meta.nvar == 5
  @test model.nls_meta.nequ == 202
  @test all(model.meta.x0 .== 1)
  @test length(findall(x -> x .!= 0, sol)) == 2
end
