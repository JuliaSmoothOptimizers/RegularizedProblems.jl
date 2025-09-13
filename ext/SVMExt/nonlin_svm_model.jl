function tan_data_train(args...)
  #load data
  A, b = MLDatasets.MNIST(split = :train)[:]
  A, b = generate_data(A, b, args...)
  return A, b
end

function tan_data_test(args...)
  A, b = MLDatasets.MNIST(split = :test)[:]
  A, b = generate_data(A, b, args...)
  return A, b
end

function generate_data(A, b, digits::Tuple{Int,Int} = (1, 7), switch::Bool = false)
  length(digits) == 2 || error("please supply two digits only")
  digits[1] != digits[2] || error("please supply two different digits")
  all(0 .≤ digits .≤ 9) || error("please supply digits from 0 to 9")
  ind = findall(x -> x ∈ digits, b)
  #reshape to matrix
  A = reshape(A, size(A, 1) * size(A, 2), size(A, 3)) ./ 255

  #get 0s and 1s
  b = b[ind]
  b[b .== digits[2]] .= -1
  A = convert(Array{Float64,2}, A[:, ind])
  if switch
    p = randperm(length(b))[1:Int(floor(length(b)/3))]
    b = b[p]
    A = A[:, p]
  end
  return A, b
end

"""
    nlp_model, nls_model, resid!, sol = svm_model(args...)

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

An instance of an `NLPModel` that represents the complete SVM problem in NLP form, and
an instance of an `NLSModel` that represents the nonlinear least squares in nonlinear least-squares form.
"""
function svm_model(A, b)
  Ahat = Diagonal(b) * A'
  r = zeros(size(Ahat, 1))
  tmp = similar(r)

  function resid!(r, x)
    mul!(r, Ahat, x)
    r .= 1 .- tanh.(r)
    r
  end
  function jacv!(Jv, x, v)
    mul!(r, Ahat, x)
    mul!(Jv, Ahat, v)
    Jv .= -((sech.(r)) .^ 2) .* Jv
  end
  function jactv!(Jtv, x, v)
    mul!(r, Ahat, x)
    tmp .= sech.(r) .^ 2
    tmp .*= v
    tmp .*= -1
    mul!(Jtv, Ahat', tmp)
  end
  function obj(x)
    resid!(r, x)
    dot(r, r) / 2
  end
  function grad!(g, x)
    mul!(r, Ahat, x)
    tmp .= (sech.(r)) .^ 2
    tmp .*= (1 .- tanh.(r))
    tmp .*= -1
    mul!(g, Ahat', tmp)
    g
  end

  FirstOrderModel(obj, grad!, ones(size(A, 1)), name = "Nonlinear-SVM"),
  FirstOrderNLSModel(resid!, jacv!, jactv!, size(b, 1), ones(size(A, 1))),
  b
end

RegularizedProblems.svm_train_model(args...) = svm_model(tan_data_train(args...)...)
RegularizedProblems.svm_test_model(args...) = svm_model(tan_data_test(args...)...)
