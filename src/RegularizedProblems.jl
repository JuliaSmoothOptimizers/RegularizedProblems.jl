module RegularizedProblems

using LinearAlgebra
using Random, Requires
using NLPModels
using Distributions

include("types.jl")
include("bpdn_model.jl")
include("nnmf_model.jl")

function __init__()
  @require ADNLPModels = "54578032-b7ea-4c30-94aa-7cbd1cce6c9a" begin
    @require DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa" begin
      include("fh_model.jl")
    end
  end
end

end
