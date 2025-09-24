# This test is skipped on FreeBSD due to issues with SciMLSensitivity and Enzyme packages

if !Sys.isfreebsd()
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
