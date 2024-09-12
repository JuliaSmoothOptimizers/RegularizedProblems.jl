module RegularizedProblems

using LinearAlgebra, SparseArrays
using Random, Requires
using NLPModels
using Distributions, Noise
using Images, FFTW, Wavelets

include("utils.jl")
include("types.jl")
include("bpdn_model.jl")
include("lrcomp_model.jl")
include("matrand_model.jl")
include("group_lasso_model.jl")
include("nnmf.jl")
include("denoising_model.jl")

function __init__()
  @require ProximalOperators = "a725b495-10eb-56fe-b38b-717eba820537" begin
    include("testset_bpdn.jl")
    include("testset_lrcomp.jl")
    include("testset_matrand.jl")
    include("testset_nnmf.jl")
  end
  @require ShiftedProximalOperators = "d4fd37fa-580c-4e43-9b30-361c21aae263" begin
    include("testset_group_lasso.jl")
  end
  @require ADNLPModels = "54578032-b7ea-4c30-94aa-7cbd1cce6c9a" begin
    @require DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa" begin
      include("fh_model.jl")
      @require ProximalOperators = "a725b495-10eb-56fe-b38b-717eba820537" begin
        include("testset_fh.jl")
      end
    end
  end
  @require MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458" begin
    include("nonlin_svm_model.jl")
    @require ProximalOperators = "a725b495-10eb-56fe-b38b-717eba820537" begin
      @require ShiftedProximalOperators = "d4fd37fa-580c-4e43-9b30-361c21aae263" begin
        include("testset_svm.jl")
      end
    end
  end
  @require QuadraticModels = "f468eda6-eac5-11e8-05a5-ff9e497bcd19" begin
    include("qp_rand_model.jl")
  end
end

end
