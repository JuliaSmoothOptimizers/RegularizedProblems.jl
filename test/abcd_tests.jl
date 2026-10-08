using MAT, ProximalOperators, Random, SparseArrays

@testset "Lasso model" begin
  Random.seed!(1234)
  m, n = 30, 20
  A = sprandn(m, n, 0.3)
  b = randn(m)
  model, nls_model = lasso_model(A, b)
  test_objectives(model, nls_model)
  @test typeof(model) <: NLPModel
  @test typeof(nls_model) <: NLSModel
  @test model.meta.nvar == n
  @test nls_model.nls_meta.nequ == m
  @test all(model.meta.x0 .== 0)
  @test !has_bounds(model)
  @test !has_bounds(nls_model)

  x, y, v = randn(n), randn(n), randn(n)
  @test obj(model, x) ≈ norm(A * x - b)^2 / 2
  @test grad(model, x) ≈ A' * (A * x - b)
  f, g = objgrad(model, x)
  @test f ≈ norm(A * x - b)^2 / 2
  @test g ≈ A' * (A * x - b)
  @test hprod(model, x, v) ≈ A' * (A * v)
  @test hprod(model, x, v; obj_weight = 2.0) ≈ 2 * (A' * (A * v))
  @test hess_op(model, x) * v ≈ A' * (A * v)
  # the cached residual must follow the evaluation point
  @test obj(model, y) ≈ norm(A * y - b)^2 / 2
  @test grad(model, x) ≈ A' * (A * x - b)
  x .+= 1
  @test obj(model, x) ≈ norm(A * x - b)^2 / 2
  @test grad(model, x) ≈ A' * (A * x - b)

  model, nls_model = lasso_model(Matrix(A), b; bounds = true)
  test_objectives(model, nls_model)
  @test get_name(model) == "Lasso-nonneg"
  @test has_bounds(model)
  @test all(model.meta.lvar .== 0)
  @test all(model.meta.uvar .== Inf)
  @test has_bounds(nls_model)
  @test all(nls_model.meta.lvar .== 0)
  @test all(nls_model.meta.uvar .== Inf)

  @test_throws DimensionMismatch lasso_model(A, randn(m + 1))
end

@testset "Logistic model" begin
  Random.seed!(1234)
  m, n = 40, 15
  A = sprandn(m, n, 0.4)
  b = rand([-1.0, 1.0], m)
  model = logistic_model(A, b)
  @test typeof(model) <: NLPModel
  @test model.meta.nvar == n
  @test all(model.meta.x0 .== 0)
  @test !has_bounds(model)

  σ(t) = 1 / (1 + exp(-t))
  f_ref(x) = sum(log1p.(exp.(-b .* (A * x))))
  g_ref(x) = A' * (-b .* σ.(-b .* (A * x)))
  function H_ref(x)
    s = σ.(-b .* (A * x))
    return A' * Diagonal(b .^ 2 .* s .* (1 .- s)) * A
  end

  x, y, v = randn(n), randn(n), randn(n)
  @test obj(model, x) ≈ f_ref(x)
  @test grad(model, x) ≈ g_ref(x)
  f, g = objgrad(model, x)
  @test f ≈ f_ref(x)
  @test g ≈ g_ref(x)
  @test hprod(model, x, v) ≈ H_ref(x) * v
  @test hprod(model, x, v; obj_weight = 2.0) ≈ 2 * (H_ref(x) * v)
  @test hess_op(model, x) * v ≈ H_ref(x) * v
  # the cached Hessian weights must follow the evaluation point
  @test hprod(model, y, v) ≈ H_ref(y) * v
  @test hprod(model, x, v) ≈ H_ref(x) * v
  @test obj(model, y) ≈ f_ref(y)
  @test grad(model, x) ≈ g_ref(x)

  # central finite differences
  d = randn(n)
  ε = 1.0e-6
  fd = (obj(model, x + ε * d) - obj(model, x - ε * d)) / (2ε)
  @test isapprox(fd, dot(g_ref(x), d), rtol = 1.0e-5)

  # values at zero: f(0) = m log 2 and ∇f(0) = -Aᵀb / 2
  @test obj(model, zeros(n)) ≈ m * log(2)
  @test grad(model, zeros(n)) ≈ -(A' * b) / 2

  # no overflow for large arguments
  @test isfinite(obj(model, 1.0e4 * x))
  @test all(isfinite, grad(model, 1.0e4 * x))
  @test all(isfinite, hprod(model, 1.0e4 * x, v))

  # labels equal to zero are allowed (they contribute the constant log 2)
  b0 = copy(b)
  b0[1] = 0
  model0 = logistic_model(Matrix(A), b0)
  @test obj(model0, x) ≈ sum(log1p.(exp.(-b0 .* (A * x))))

  @test_throws DimensionMismatch logistic_model(A, randn(m + 1))
end

@testset "ABCD registry" begin
  @test length(abcd_problems()) == 49
  @test length(abcd_problems(:lasso)) == 49
  @test length(abcd_problems(:lasso_tuning)) == 18
  @test length(abcd_problems(:lasso_test)) == 31
  @test length(abcd_problems(:logistic)) == 35
  @test length(abcd_problems(:logistic_all)) == 49
  @test length(abcd_problems(:logistic_tuning)) == 12
  @test length(abcd_problems(:logistic_test)) == 23
  @test length(abcd_problems(:nonneg_lasso)) == 12
  @test sort(vcat(abcd_problems(:lasso_tuning), abcd_problems(:lasso_test))) ==
        sort(abcd_problems(:lasso))
  @test sort(vcat(abcd_problems(:logistic_tuning), abcd_problems(:logistic_test))) ==
        sort(abcd_problems(:logistic))
  @test issubset(abcd_problems(:logistic), abcd_problems(:logistic_all))
  @test_throws ArgumentError abcd_problems(:foo)

  for set ∈ (:lasso, :logistic_all, :nonneg_lasso), name ∈ abcd_problems(set)
    info = abcd_problem_info(name)
    @test info.name == name
    @test info.file == name * ".mat"
  end

  info = abcd_problem_info("SR10")
  @test info.kind == :lasso
  @test info.folder == "Data-Lasso"
  @test info.dataset == "w2a"
  @test (info.nrow, info.ncol) == (3470, 293)
  info = abcd_problem_info("SR10"; kind = :logistic)
  @test info.name == "SRlog10"
  @test info.kind == :logistic
  @test info.folder == "Data-Logistic"
  @test abcd_problem_info(:SClog8).name == "SClog8"
  @test abcd_problem_info("NN3").kind == :nonneg_lasso
  @test abcd_problem_info("NN3").folder == "Data-Non-Negative-Lasso"
  @test_throws ArgumentError abcd_problem_info("SR25")
  @test_throws ArgumentError abcd_problem_info("NN13")
  @test_throws ArgumentError abcd_problem_info("SRlog10"; kind = :lasso)
  @test_throws ArgumentError abcd_problem_info("NN3"; kind = :logistic)
end

@testset "ABCD loaders (synthetic files)" begin
  Random.seed!(0)
  m, n = 25, 10
  A = sprandn(m, n, 0.5)
  b = randn(m)
  labels = rand([-1.0, 1.0], m)
  λ = norm(A' * b, Inf) / 10
  λlog = norm(A' * labels, Inf) / 20

  dir = mktempdir()
  for folder ∈ ("Data-Lasso", "Data-Logistic", "Data-Non-Negative-Lasso")
    mkpath(joinpath(dir, folder))
  end
  matwrite(
    joinpath(dir, "Data-Lasso", "SR10.mat"),
    Dict("A" => A, "b" => b, "lambda" => λ, "ftarget" => 1.5),
  )
  # some original files store integer labels and targets
  matwrite(
    joinpath(dir, "Data-Lasso", "SC8.mat"),
    Dict("A" => A, "b" => Int16.(sign.(b)), "lambda" => λ, "ftarget" => Int32(3)),
  )
  matwrite(
    joinpath(dir, "Data-Logistic", "SRlog10.mat"),
    Dict("A" => A, "b" => labels, "lambdalog" => λlog, "flogtarget" => 2.5),
  )
  matwrite(
    joinpath(dir, "Data-Non-Negative-Lasso", "NN1.mat"),
    Dict("A" => A, "b" => b, "lambda" => λ, "ftarget" => 3.5),
  )

  data = abcd_data("SR10"; datadir = dir)
  @test data.name == "SR10"
  @test data.kind == :lasso
  @test data.A isa SparseMatrixCSC{Float64, Int}
  @test data.A == A
  @test data.b isa Vector{Float64}
  @test data.b == b
  @test data.λ == λ
  @test data.ftarget == 1.5

  data = abcd_data(:SC8; datadir = dir)
  @test data.b isa Vector{Float64}
  @test data.b == sign.(b)
  @test data.ftarget === 3.0

  model, nls_model, data = abcd_lasso_model("SR10"; datadir = dir)
  test_objectives(model, nls_model)
  @test get_name(model) == "SR10"
  @test !has_bounds(model)
  @test grad(model, zeros(n)) ≈ -(A' * b)
  @test norm(grad(model, zeros(n)), Inf) / 10 ≈ data.λ

  model, nls_model, data = abcd_lasso_model("NN1"; datadir = dir)
  test_objectives(model, nls_model)
  @test data.kind == :nonneg_lasso
  @test has_bounds(model)
  @test all(model.meta.lvar .== 0)
  @test has_bounds(nls_model)

  for name ∈ ("SR10", "SRlog10", :SRlog10)
    model, data = abcd_logistic_model(name; datadir = dir)
    @test get_name(model) == "SRlog10"
    @test data.kind == :logistic
    @test data.b == labels
    @test data.λ == λlog
    @test data.ftarget == 2.5
    @test obj(model, zeros(n)) ≈ m * log(2)
    @test norm(grad(model, zeros(n)), Inf) / 10 ≈ data.λ
  end

  @test_throws ArgumentError abcd_lasso_model("SRlog10"; datadir = dir)
  @test_throws ArgumentError abcd_data("SR99"; datadir = dir)
  @test_throws ErrorException abcd_data("SR11"; datadir = dir)  # missing file

  withenv("ABCD_DATA_DIR" => nothing) do
    @test_throws ErrorException abcd_datadir()
  end
  withenv("ABCD_DATA_DIR" => dir) do
    @test abcd_datadir() == dir
    @test abcd_data("SR10").A == A
  end

  rmodel, rnls_model, data = setup_abcd_lasso_l1("SR10"; datadir = dir)
  @test rmodel isa RegularizedNLPModel
  @test rnls_model isa RegularizedNLSModel
  @test get_name(rmodel) == "SR10/NormL1"
  @test rmodel.h(ones(n)) ≈ n * data.λ
  @test obj(rmodel, zeros(n)) ≈ norm(b)^2 / 2

  rmodel, data = setup_abcd_logistic_l1("SR10"; datadir = dir, λ = 1.0)
  @test rmodel isa RegularizedNLPModel
  @test rmodel.h(ones(n)) ≈ n
  @test obj(rmodel, zeros(n)) ≈ m * log(2)
end

# Optional checks against the original files; run them with
#   ENV["ABCD_DATA_DIR"] = "/path/to/Data-Files"
if isdir(get(ENV, "ABCD_DATA_DIR", ""))
  @testset "ABCD loaders (original files)" begin
    model, nls_model, data = abcd_lasso_model("SR10")
    @test size(data.A) == (3470, 293)
    @test data.λ ≈ norm(grad(model, zeros(293)), Inf) / 10
    @test data.ftarget ≈ 1069
    model, data = abcd_logistic_model("SR10")
    @test data.λ ≈ norm(grad(model, zeros(293)), Inf) / 10
    @test data.ftarget ≈ 1551
    data = abcd_data("SC10")  # MATLAB v5 file with Int16 labels
    @test size(data.A) == (300, 7847)
    @test data.b isa Vector{Float64}
    @test data.ftarget ≈ 114.7
    model, nls_model, data = abcd_lasso_model("NN1")
    @test has_bounds(model)
    @test size(data.A) == (1033, 320)
    @test data.ftarget ≈ 1.098e7
  end
end
