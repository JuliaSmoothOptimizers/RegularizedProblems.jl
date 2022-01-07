export tanh_train_model, tanh_test_model#, tan_nls_model
using MLDatasets, Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold
function tan_data_train()
  #load data
  A, b = MNIST.traindata();
  ind = findall(x -> x == 0 || x == 1, b);
  #reshape to matrix
  A = reshape(A,size(A,1)*size(A,2), size(A,3))./255;

  #get 0s and 1s
  b = b[ind]
  b[b.==0] .= -1
  A = A[:, ind]

  x0 = ones(size(A,1))
  A, b, x0
end

function tan_data_test()
  A0, b0 = MNIST.testdata();
  ind = findall(x -> x == 0 || x == 1, b)
  A0 = reshape(A0,size(A0,1)*size(A0,2), size(A0,3))./255

  #get 0s and 1s
  b0 = b0[ind]
  b0[b0.==0] .= -1
  A0 = A0[:, ind]

  x0 = ones(size(A0,1))
  A0, b0, x0
end

"""
    model, sol = bpdn_model(args...)
    model, sol = bpdn_model(compound = 1, args...)

Return an instance of an `NLPModel` representing the basis-pursuit denoise
problem, i.e., the under-determined linear least-squares objective

   f(x) = ∑ 1 - tanh(bᵢ⋅⟨aᵢ, x⟩),
   h(x) = ‖ ⋅ ‖

where A has orthonormal rows and b = A * x̄ + ϵ, x̄ is sparse and ϵ is a noise
vector following a normal distribution with mean zero and standard deviation σ.

## Arguments

* `m :: Int`: the number of rows of A
* `n :: Int`: the number of columns of A (with `n` ≥ `m`)
* `k :: Int`: the number of nonzero elements in x̄
* `noise :: Float64`: noise amount ϵ (default: 0.01).

The second form calls the first form with arguments

    m = 200 * compound
    n = 512 * compound
    k =  10 * compound

## Return Value

An instance of a `FirstOrderModel` that represents the basis-pursuit denoise problem
and the exact solution x̄.
"""
function tanh_train_model()
  A, b, x0 = tan_data_train()
  r = zeros(size(A,2))

  function resid!(r, x)
    mul!(r, A', x)
    r .= 1 .- tanh.(b .* r)
    r
  end

  function obj(x)
    resid!(r, x)
    # dot(r, r) / 2 # can switch back
    sum(r)
  end

  function grad!(g, x)
    mul!(r, A', x)
    r .= (1 .- tanh.(b .* r).^2) .* b
    mul!(g, -A, r)
    g
  end

  FirstOrderModel(obj, grad!, ones(size(x0)), name = "MNIST-tanh"), b
end

function tanh_test_model()
  A, b, x0 = tan_data_test()
  r = zeros(size(A,2))

  function resid!(r, x)
    mul!(r, A', x)
    r .= 1 .- tanh.(b .* r)
    r
  end

  function obj(x)
    resid!(r, x)
    # dot(r, r) / 2 # can switch back
    sum(r)
  end

  function grad!(g, x)
    mul!(r, A', x)
    r .= (1 .- tanh.(b .* r).^2) .* b
    mul!(g, -A, r)
    g
  end

  FirstOrderModel(obj, grad!, ones(size(x0)), name = "MNIST-tanh"), b
end

function getdata()
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

  # Loading Dataset
  xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
  xtest, ytest = MLDatasets.MNIST.testdata(Float32)

  # Reshape Data in order to flatten each image into a linear array
  xtrain = Flux.flatten(xtrain)
  xtest = Flux.flatten(xtest)

  # One-hot-encode the labels
  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

  return xtrain, xtest, ytrain, ytest
end

function build_model(; imgsize=(28,28,1), nclasses=10)
  return Chain( Dense(prod(imgsize), 32, relu),
                Dense(32, nclasses))
end

function nn_test_model()
  trainx, testx, trainy, testy = getdata()

  model = build_model()

  ps = Flux.params(model)




end

"""
    model, sol = bpdn_nls_model(args...)
    model, sol = bpdn_nls_model(compound = 1, args...)

Return an instance of a `FirstOrderNLSModel` that represents the basis-pursuit
denoise problem explicitly as a least-squares problem and the exact solution x̄.

See the documentation of `bpdn_model()` for more information and a
description of the arguments.
"""
# function bpdn_nls_model(args...)
#   A, b, b0, x0 = bpdn_data(args...)
#   r = similar(b)

#   function resid!(r, x)
#     mul!(r, A, x)
#     r .-= b
#     r
#   end

#   jprod_resid!(Jv, x, v) = mul!(Jv, A, v)
#   jtprod_resid!(Jtv, x, v) = mul!(Jtv, A', v)

#   FirstOrderNLSModel(resid!, jprod_resid!, jtprod_resid!, size(A, 1), zero(x0), name = "BPDN-LS"),
#   x0
# end
