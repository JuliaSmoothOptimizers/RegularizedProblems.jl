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
  @test typeof(hess_op(rmodel_lbfgs, model.meta.x0)) <: LBFGSOperator
  B_init = Matrix(hess_op(rmodel_lbfgs, model.meta.x0))
  push!(rmodel_lbfgs.model, model.meta.x0, grad(model, model.meta.x0))
  reset!(rmodel_lbfgs)
  @test Matrix(hess_op(rmodel_lbfgs, model.meta.x0)) == B_init
  @test neval_grad(rmodel_lbfgs) == 0
end

@testset "ShiftedProximableQuadraticNLPModel" begin
  for bounds in (false, true)
    model, nls_model, x0 = bpdn_model(1, bounds = bounds)
    s = ones(get_nvar(model))
    h = NormL0(1.0)
    rmodel = RegularizedNLPModel(LBFGSModel(model), h)
    subproblem = ShiftedProximableQuadraticNLPModel(rmodel, x0)
    
    @test get_nvar(subproblem) == get_nvar(rmodel)

    obj(subproblem, s)
    @test neval_obj(subproblem) == neval_obj(subproblem.model)

    @test get_sigma(subproblem) == 0.0
    update_sigma!(subproblem, 1.0)
    @test get_sigma(subproblem) == 1.0

    @test obj(subproblem, s; skip_sigma = false) == obj(subproblem, s; skip_sigma = true) + 0.5 * dot(s, s)
    @test obj(subproblem, s; cauchy = true) == dot(subproblem.model.data.c, s) + 0.5 * dot(s, s) + subproblem.h(s)
    @test obj(subproblem, s; cauchy = true, skip_sigma = true) == dot(subproblem.model.data.c, s) + subproblem.h(s)

    update_sigma!(subproblem, 0.0)
    @test obj(subproblem, s; skip_sigma = false) == obj(subproblem, s; skip_sigma = true)
    @test obj(subproblem, s; cauchy = true, skip_sigma = false) == obj(subproblem, s; cauchy = true, skip_sigma = true)

    x = randn(get_nvar(model))
    shift!(subproblem, x)
    @test subproblem.model.data.c == grad(model, x)
    @test typeof(subproblem.model.data.H) <: LBFGSOperator
    if bounds == true
      @test all(subproblem.h.l .== model.meta.lvar - x)
      @test all(subproblem.h.u .== model.meta.uvar - x)
    end

    # Test allocations
    @test (@allocated obj(subproblem, s)) == 0
    @test (@allocated obj(subproblem, s; skip_sigma = true)) == 0
    @test (@allocated obj(subproblem, s; cauchy = true)) == 0
    @test (@allocated obj(subproblem, s; cauchy = true, skip_sigma=true)) == 0
    @test (@allocated shift!(subproblem, x)) == 0
  end
end

@testset "Problem combos" begin
  # Test that we can at least instantiate the models
  rnlp, rnls = setup_bpdn_l0()
  @test isa(rnlp, RegularizedNLPModel)
  @test isa(rnls, RegularizedNLSModel)
end
