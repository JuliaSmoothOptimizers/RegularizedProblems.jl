using LinearAlgebra, Test
using ADNLPModels,
  DifferentialEquations,
  ManualNLPModels,
  MLDatasets,
  NLPModels,
  ProximalOperators,
  QuadraticModels
using RegularizedProblems

function test_well_defined(model, nls_model, sol)
  @test typeof(model) <: NLPModel
  @test typeof(sol) == typeof(model.meta.x0)
  @test typeof(nls_model) <: NLSModel
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

#=
Don't add your tests to runtests.jl. Instead, create files named

    test-title-for-my-test.jl

The file will be automatically included inside a `@testset` with title "Title For My Test".
=#
for (root, dirs, files) in walkdir(@__DIR__)
  for file in files
    if isnothing(match(r"^test-.*\.jl$", file))
      continue
    end
    title = titlecase(replace(splitext(file[6:end])[1], "-" => " "))
    @testset "$title" begin
      include(file)
    end
  end
end
