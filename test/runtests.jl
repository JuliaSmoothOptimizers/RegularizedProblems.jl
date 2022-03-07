using Test
using ADNLPModels, DifferentialEquations
using RegularizedProblems

@testset "BPDN" begin
  model, nls_model, sol = bpdn_model()
  @test typeof(model) <: FirstOrderModel
  @test typeof(sol) == Vector{Float64}
  @test model.meta.nvar == 512
  @test all(model.meta.x0 .== 0)
  @test length(findall(x -> x .!= 0, sol)) == 10

  @test typeof(nls_model) <: FirstOrderNLSModel
  @test nls_model.meta.nvar == 512
  @test nls_model.nls_meta.nequ == 200
  @test all(nls_model.meta.x0 .== 0)
end

@testset "FH" begin
  model, nls_model, sol = fh_model()
  @test typeof(model) <: ADNLPModel
  @test typeof(sol) == Vector{Float64}
  @test model.meta.nvar == 5
  @test all(model.meta.x0 .== 1)
  @test length(findall(x -> x .!= 0, sol)) == 2

  @test typeof(nls_model) <: ADNLSModel
  @test nls_model.meta.nvar == 5
  @test nls_model.nls_meta.nequ == 202
  @test all(nls_model.meta.x0 .== 1)
end
