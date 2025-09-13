module RegularizedProblems

using LinearAlgebra, SparseArrays
using Random
using ManualNLPModels, NLPModels, ShiftedProximalOperators
using Distributions, Noise, ProximalOperators


include("utils.jl")
include("types.jl")
include("bpdn_model.jl")
include("lrcomp_model.jl")
include("matrand_model.jl")
include("group_lasso_model.jl")
include("nnmf.jl")

include("testset_bpdn.jl")
include("testset_group_lasso.jl")
include("testset_lrcomp.jl")
include("testset_matrand.jl")
include("testset_nnmf.jl")

# define generic functions that will be overloaded in extensions
# FHExt
export fh_model, setup_fh_l0, setup_fh_l1
function fh_model end
function setup_fh_l0 end
function setup_fh_l1 end

# QPRandExt
export qp_rand_model, setup_qp_rand_l1
function qp_rand_model end
function setup_qp_rand_l1 end

# SVMExt
export svm_train_model,
  svm_test_model,
  setup_svm_train_lhalf,
  setup_svm_test_lhalf,
  setup_svm_train_l0,
  setup_svm_test_l0,
  setup_svm_train_l1,
  setup_svm_test_l1
function svm_train_model end
function svm_test_model end
function setup_svm_train_lhalf end
function setup_svm_test_lhalf end
function setup_svm_train_l0 end
function setup_svm_test_l0 end
function setup_svm_train_l1 end
function setup_svm_test_l1 end

# function __init__()
#   # @require ADNLPModels = "54578032-b7ea-4c30-94aa-7cbd1cce6c9a" begin
#   #   @require DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa" begin
#   #     include("fh_model.jl")
#   #     include("testset_fh.jl")
#   #   end
#   # end
#   # @require MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458" begin
#   #   include("nonlin_svm_model.jl")
#   #   include("testset_svm.jl")
#   # end
#   # @require QuadraticModels = "f468eda6-eac5-11e8-05a5-ff9e497bcd19" begin
#   #   include("qp_rand_model.jl")
#   # end
# end

end
