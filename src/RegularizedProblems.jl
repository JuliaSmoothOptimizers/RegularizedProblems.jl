module RegularizedProblems

using LinearAlgebra
using Random, Requires
using NLPModels
using Distributions, Noise

include("utils.jl")
include("types.jl")
include("bpdn_model.jl")
include("lrcomp_model.jl")
include("matrand_model.jl")
include("group_lasso_model.jl")

function __init__()
  @require ADNLPModels = "54578032-b7ea-4c30-94aa-7cbd1cce6c9a" begin
    @require DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa" begin
      include("fh_model.jl")
    end
  end
  @require MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458" begin
    include("nonlin_svm_model.jl")
  end
end

end
