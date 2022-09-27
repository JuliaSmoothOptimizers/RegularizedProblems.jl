export lrcomp_model

function lrcomp_data(m::Int, n::Int; T::DataType = Float64)
  A = Array(rand(T, (m, n)))
  A
end

function lrcomp_model(m::Int, n::Int; T::DataType = Float64)
  A = lrcomp_data(m, n, T = T)
  r = vec(similar(A))

  function resid!(r, x)
    for i in eachindex(A)
      r[i] = x[i] - A[i]
    end
    r
  end

  function jprod_resid!(Jv, x, v)
    Jv .= v
    Jv
  end

  function obj(x)
    resid!(r, x)
    dot(r, r) / 2
  end

  grad!(r, x) = resid!(r, x)

  x0 = rand(T, m * n)
  FirstOrderModel(obj, grad!, x0, name = "LRCOMP"),
  FirstOrderNLSModel(resid!, jprod_resid!, jprod_resid!, m * n, x0, name = "LRCOMP-LS"),
  vec(A)
end
