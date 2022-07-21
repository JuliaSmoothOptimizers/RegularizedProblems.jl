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

@testset "SVM-Train" begin
  nlp_train,nls_train,sol = svm_train_model()
  @test typeof(nlp_train) <: ADNLPModel
  @test typeof(nls_train) <: ADNLSModel
  @test typeof(sol) == Vector{Float64}
  ### below is TBD
  # @test model.meta.nvar == 5
  # @test model.nls_meta.nequ == 202
  # @test all(model.meta.x0 .== 1)
  # @test length(findall(x -> x .!= 0, sol)) == 2
end

@testset "SVM-Test" begin
  nlp_test,nls_test,sol = svm_test_model()
  @test typeof(nlp_test) <: ADNLPModel
  @test typeof(nls_test) <: ADNLSModel
  @test typeof(sol) == Vector{Float64}
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