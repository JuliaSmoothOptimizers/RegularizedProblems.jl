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
include("nnmf.jl")

# move it back to fh_model.jl once DifferentialEquations precompilation is fixed?
"""
    fh_model(; kwargs...)

Return an instance of an `NLPModel` and an instance of an `NLSModel` representing
the same Fitzhugh-Nagumo problem, i.e., the over-determined nonlinear
least-squares objective

   ½ ‖F(x)‖₂²,

where F: ℝ⁵ → ℝ²⁰² represents the fitting error between a simulation of the
Fitzhugh-Nagumo model with parameters x and a simulation of the Van der Pol
oscillator with fixed, but unknown, parameters.

Requires ADNLPModels.jl and DifferentialEquations.jl to be imported.

## Keyword Arguments

All keyword arguments are passed directly to the `ADNLPModel` (or `ADNLSModel`)
constructure, e.g., to set the automatic differentiation backend.

## Return Value

An instance of an `ADNLPModel` that represents the Fitzhugh-Nagumo problem, an instance
of an `ADNLSModel` that represents the same problem, and the exact solution.
"""
function fh_model end

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
