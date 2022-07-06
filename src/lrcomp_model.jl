export lrcomp_model

function lrcomp_data(m::Int, n::Int; T::DataType = Float64)
  A = Array(rand(T, (m, n)))
  A
end

function lrcomp_model(m::Int, n::Int)
  A = lrcomp_data(m, n)
  r = vec(similar(A))

  function resid!(r, x)
    for i in eachindex(A)
      r[i] = x[i] - A[i]
    end
    r
  end

  function obj(x)
    resid!(r, x)
    dot(r, r) / 2
  end

  grad!(r, x) = resid!(r, x)

  FirstOrderModel(obj, grad!, rand(T, m * n), name = "LRCOMP")
end
