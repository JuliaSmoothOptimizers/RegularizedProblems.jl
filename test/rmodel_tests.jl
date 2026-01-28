using ProximalOperators

@testset "RegularizedNLPModel" begin
  model, nls_model, x0 = bpdn_model(1)
  h = NormL0(1.0)
  rmodel = RegularizedNLPModel(model, h)
  bpdn_name = get_name(model)
  @test get_name(rmodel) == bpdn_name * "/NormL0"
  @test get_nvar(rmodel) == get_nvar(model)
  obj(model, model.meta.x0)
  @test neval_obj(rmodel) == neval_obj(model)
  rlsmodel = RegularizedNLSModel(nls_model, h)
  bpdn_ls_name = get_name(nls_model)
  @test get_name(rlsmodel) == bpdn_ls_name * "/NormL0"
  @test get_nequ(rlsmodel) == get_nequ(nls_model)
  obj(nls_model, nls_model.meta.x0)
  @test neval_obj(rlsmodel) == neval_obj(nls_model)
  rmodel_lbfgs = RegularizedNLPModel(LBFGSModel(model), h)
  @test typeof(hess_op(rmodel_lbfgs)) <: LBFGSOperator
end

@testset "Problem combos" begin
  # Test that we can at least instantiate the models
  rnlp, rnls = setup_bpdn_l0()
  @test isa(rnlp, RegularizedNLPModel)
  @test isa(rnls, RegularizedNLSModel)
end
