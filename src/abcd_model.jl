export abcd_datadir, abcd_data, abcd_lasso_model, abcd_logistic_model

const _ABCD_URL = "https://drive.google.com/drive/folders/1yUqNvwDazKED-Umfe5O95WP-JyedkLnm"

"""
    dir = abcd_datadir()

Return the directory containing the data of Lopes, Santos and Silva (2019), i.e., the folder
in which `Data-Files.zip` was extracted. It must contain the subfolders `Data-Lasso`,
`Data-Logistic` and `Data-Non-Negative-Lasso` (only those needed are required).

The directory is read from the environment variable `ABCD_DATA_DIR`, e.g.,

    ENV["ABCD_DATA_DIR"] = joinpath(homedir(), "Downloads", "Data-Files")

The data can be downloaded from $(_ABCD_URL).
"""
function abcd_datadir()
  dir = get(ENV, "ABCD_DATA_DIR", "")
  isempty(dir) && error(
    "the location of the ABCD data is unknown. Download and extract `Data-Files.zip` from " *
    "$(_ABCD_URL), then set ENV[\"ABCD_DATA_DIR\"] to the folder that contains `Data-Lasso`, " *
    "`Data-Logistic` and `Data-Non-Negative-Lasso`, or pass the keyword argument `datadir`.",
  )
  isdir(dir) || error("ENV[\"ABCD_DATA_DIR\"] = \"$dir\" is not a directory")
  return dir
end

_abcd_scalar(x) = Float64(x isa AbstractArray ? only(x) : x)

function _abcd_read(file, path, var)
  haskey(file, var) || error("variable `$var` not found in $path")
  return read(file, var)
end

"""
    data = abcd_data(name; datadir = abcd_datadir(), kind = nothing)

Read the problem `name` of the data set of Lopes, Santos and Silva (2019) from its MAT file
and return a named tuple with fields

* `name`, `kind`: as in `abcd_problem_info`;
* `A::SparseMatrixCSC{Float64, Int}` (or `Matrix{Float64}` if stored dense): the data matrix;
* `b::Vector{Float64}`: the right-hand side (Lasso) or labels (logistic regression);
* `λ::Float64`: the regularization parameter used in the paper, λ = 0.1 ‖∇f(0)‖∞;
* `ftarget::Float64`: the target objective value used as stopping criterion in the paper, an upper
  bound on the optimal value of f + λ‖⋅‖₁ with relative error of order 10⁻⁴;
* `path`: the file that was read.

This function is available after `using MAT`.
See `abcd_problems` for valid names and `abcd_problem_info` for the meaning of `kind`.
"""
function abcd_data(
  name::Union{AbstractString, Symbol};
  datadir::AbstractString = abcd_datadir(),
  kind::Union{Symbol, Nothing} = nothing,
)
  info = abcd_problem_info(name; kind = kind)
  path = joinpath(datadir, info.folder, info.file)
  isfile(path) || error(
    "data file $path not found; `datadir` should be the folder in which `Data-Files.zip` " *
    "was extracted (available from $(_ABCD_URL))",
  )
  λvar, fvar = info.kind == :logistic ? ("lambdalog", "flogtarget") : ("lambda", "ftarget")
  file = MAT.matopen(path)
  A, b, λ, ftarget = try
    (
      _abcd_read(file, path, "A"),
      _abcd_read(file, path, "b"),
      _abcd_read(file, path, λvar),
      _abcd_read(file, path, fvar),
    )
  finally
    close(file)
  end
  # Some files were saved by older MATLAB versions with integer labels/targets.
  A = A isa AbstractSparseMatrix ? convert(SparseMatrixCSC{Float64, Int}, A) : Matrix{Float64}(A)
  b = b isa AbstractArray ? Vector{Float64}(vec(b)) : [Float64(b)]
  size(A, 1) == length(b) ||
    error("inconsistent data in $path: A is $(size(A)) but b has $(length(b)) entries")
  return (
    name = info.name,
    kind = info.kind,
    A = A,
    b = b,
    λ = _abcd_scalar(λ),
    ftarget = _abcd_scalar(ftarget),
    path = path,
  )
end

"""
    model, nls_model, data = abcd_lasso_model(name; datadir = abcd_datadir())

Return the smooth part f(x) = ½‖Ax - b‖₂² of the Lasso problem `name` of Lopes, Santos and
Silva (2019) as an `NLPModel` and an `NLSModel` (see `lasso_model`), together with the data
returned by `abcd_data`.
For `name ∈ abcd_problems(:nonneg_lasso)` (`"NN1"`, …, `"NN12"`), the models include the bounds x ≥ 0.

The problem solved in the paper is

    minimize ½‖Ax - b‖₂² + λ‖x‖₁  (subject to x ≥ 0 for NN problems)

with `λ = data.λ`, starting from x = 0 and stopping as soon as the objective is below `data.ftarget`.
See `setup_abcd_lasso_l1` for the corresponding regularized models.

This function is available after `using MAT`.
"""
function abcd_lasso_model(
  name::Union{AbstractString, Symbol};
  datadir::AbstractString = abcd_datadir(),
)
  info = abcd_problem_info(name)
  info.kind == :logistic && throw(
    ArgumentError("\"$name\" is a logistic regression problem; use `abcd_logistic_model`"),
  )
  data = abcd_data(info.name; datadir = datadir)
  nonneg = data.kind == :nonneg_lasso
  model, nls_model = lasso_model(data.A, data.b; bounds = nonneg, name = data.name)
  return model, nls_model, data
end

"""
    model, data = abcd_logistic_model(name; datadir = abcd_datadir())

Return the smooth part f(x) = ∑ᵢ log(1 + exp(-bᵢ aᵢᵀx)) of the ℓ₁-regularized logistic
regression problem `name` of Lopes, Santos and Silva (2019) as an `NLPModel` (see `logistic_model`),
together with the data returned by `abcd_data`.
`name` may be given as `"SRlog10"` or `"SR10"`; both refer to the file `Data-Logistic/SRlog10.mat`.

The problem solved in the paper is

    minimize ∑ᵢ log(1 + exp(-bᵢ aᵢᵀx)) + λ‖x‖₁

with `λ = data.λ`, starting from x = 0 and stopping as soon as the objective is below `data.ftarget`.
See `setup_abcd_logistic_l1` for the corresponding regularized model.

This function is available after `using MAT`.
"""
function abcd_logistic_model(
  name::Union{AbstractString, Symbol};
  datadir::AbstractString = abcd_datadir(),
)
  info = abcd_problem_info(name; kind = :logistic)
  data = abcd_data(info.name; datadir = datadir)
  model = logistic_model(data.A, data.b; name = data.name)
  return model, data
end
