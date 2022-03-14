export tanh_train_model, tanh_test_model#, tan_nls_model

function tan_data_train()
  #load data
  A, b = MNIST.traindata()
  ind = findall(x -> x == 0 || x == 1, b)
  #reshape to matrix
  A = reshape(A,size(A,1)*size(A,2), size(A,3))./255

  #get 0s and 1s
  b = b[ind]
  b[b.==0] .= -1
  A = convert(Array{Float64, 2}, A[:, ind])
  p = randperm(length(b))[1:Int(floor(length(b)/4))]
  b = b[p]
  A = A[:, p]

  x0 = ones(size(A,1))
  A, b, x0
end

function tan_data_test()
  A0, b0 = MNIST.testdata()
  ind = findall(x -> x == 0 || x == 1, b)
  A0 = reshape(A0,size(A0,1)*size(A0,2), size(A0,3))./255

  #get 0s and 1s
  b0 = b0[ind]
  b0[b0.==0] .= -1
  A0 = convert(Array{Float64, 2}, A0[:, ind])

  x0 = ones(size(A0,1))
  A0, b0, x0
end

"""
    model, sol = tanh_train_model(args...)
    model, sol = tanh_test_model(args...)

Return an instance of an `NLPModel` representing the hyperbolic SVM
problem, i.e., the under-determined linear least-squares objective

   f(x) = ‖1 - tanh(b ⊙ ⟨A, x⟩)‖²,

where A is the data matrix with labels b = {-1, 1}ⁿ.

## Arguments

* `m :: Int`: the number of rows of A, size of b
* `n :: Int`: the number of columns of A (with `n` ≥ `m`)

With the MNIST Dataset, the dimensions are:

    m = 12665
    n = 784

## Return Value

An instance of a `FirstOrderModel` that represents the complete SVM problem in NLP form, and
an instance of `FirstOrderNLSModel` that represents the nonlinear least squares in nonlinear least squares form.
"""
function tanh_train_model()
  A, b, x0 = tan_data_train()
  Ahat = Diagonal(b)*A'
  r = zeros(size(Ahat,1))
  tmp = similar(r)

  function resid!(r, x)
    mul!(r, Ahat, x)
    r .= 1 .- tanh.(r)
    r
  end
  function resid(x)
    return 1 .- tanh.(Ahat * x)
  end

  function jacv!(Jv, x, v)
    mul!(r, Ahat, x)
    mul!(Jv, Ahat, v)
    Jv .= -((sech.(r)).^2) .* Jv
  end
  function jactv!(Jtv, x, v)
    mul!(r, Ahat, x)
    tmp .= sech.(r).^2
    tmp .*= v
    tmp .*= -1.
    mul!(Jtv, Ahat', tmp)
  end
  function obj(x)
    r = resid(x)
    dot(r, r) / 2
    # sum(r)
  end

  function grad!(g, x)
    mul!(r, Ahat, x)
    tmp .= (sech.(r)).^2
    tmp .*= (1 .- tanh.(r))
    tmp .*= -1.
    mul!(g, Ahat', tmp)
    g
  end

  FirstOrderModel(obj, grad!, ones(size(x0)), name = "MNIST-tanh"), FirstOrderNLSModel(resid!, jacv!, jactv!, size(b,1), x0), resid, obj, b
end

function tanh_test_model()
  A, b, x0 = tan_data_test()
  Ahat = Diagonal(b)*A'
  r = zeros(size(Ahat,1))
  tmp = similar(r)

  function resid!(r, x)
    mul!(r, A', x)
    r .= 1 .- tanh.(b .* r)
    r
  end

  function jacv!(Jv, x, v)
    mul!(r, Ahat, x)
    mul!(Jv, Ahat, v)
    Jv .= -((sech.(r)).^2) .* Jv
  end
  function jactv!(Jtv, x, v)
    mul!(r, Ahat, x)
    tmp .= sech.(r).^2
    tmp .*= v
    tmp .*= -1.
    mul!(Jtv, Ahat', tmp)
  end

  function obj(x)
    resid!(r, x)
    dot(r, r) / 2
    # sum(r)
  end

  function grad!(g, x)
    mul!(r, Ahat, x)
    tmp .= (sech.(r)).^2
    tmp .*= (1 .- tanh.(r))
    tmp .*= -1.
    mul!(g, Ahat', tmp)
    g
  end

  FirstOrderModel(obj, grad!, ones(size(x0)), name = "MNIST-tanh"), FirstOrderNLSModel(resid!, jacv!, jactv!, size(b,1), x0), resid, obj, b
end

function mnist_model(; kwargs...)
  tanhnlp_train, tanhnls_train, resid_train, obj_train, sol_train = tanh_train_model()
  tanhnlp_test,  tanhnls_test,  resid_test, obj_test,   sol_test  = tanh_test_model()
end
