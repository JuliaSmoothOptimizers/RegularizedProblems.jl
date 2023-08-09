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

function __init__()
  @require ADNLPModels = "54578032-b7ea-4c30-94aa-7cbd1cce6c9a" begin
    @require DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa" begin
      include("fh_model.jl")
    end
  end
  @require MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458" begin
    include("nonlin_svm_model.jl")
  end
  @require FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341" begin
    @require Wavelets = "29a6e085-ba6d-5f35-a997-948ac2efa89a" begin
      @require Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0" begin
        include("denoising_model.jl")
      end
    end
  end
end

end
