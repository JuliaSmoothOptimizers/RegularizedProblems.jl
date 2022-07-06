export lrcomp_model

function lrcomp_data(m::Int, n::Int)
  A = Array(rand(Float64,(m, n)))
  A
end

function lrcomp_model(m::Int, n::Int)
  A = lrcomp_data(m, n)
  r = vec(similar(A))

  function resid!(r, x, A)
    for i in eachindex(A)
      r[i] = x[i] - A[i]
    end
  end


  function obj(x)
    resid!(r, x, A)
    dot(r, r) / 2
  end

  function grad!(r, x)
    resid!(r, x, A) 
    r
  end

  FirstOrderModel(obj, grad!, rand(Float64, m*n), name = "LRCOMP")
end