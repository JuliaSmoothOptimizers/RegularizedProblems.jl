export nnmf_model

function nnmf_data(m::Int, n::Int, T::DataType = Float64)
  A = Array(rand(T,(n, m)))
  A
end

function nnmf_model(m::Int, n::Int, k::Int)
  A = nnmf_data(m, n)
  r = similar(A, m*n)
  WH = similar(A)
  gw = similar(A, (m,k))
  gh = similar(A, (k,n))

  function resid!(r, x)
    W = reshape_array(x[1:(m*k)], (m,k))
    H = reshape_array(x[(m*k+1):end], (k,n))
    mul!(WH, W, H)
    for i ∈ eachindex(r)
      r[i] = A[i] - WH[i]
    end
    r
  end

  function obj(x)
    resid!(r, x)
    dot(r, r) / 2
  end

  function grad!(g, x)
    resid!(r, x)
    minusR = -reshape_array(r, (m,n))
    W_T = reshape_array(x[1:(m*k)], (m,k))'
    H_T = reshape_array(x[(m*k+1):end], (k,n))'
    mul!(gw, minusR, H_T)
    mul!(gh, W_T, minusR)
    g .= vcat(vec(gw), vec(gh))
    g
  end

  FirstOrderModel(obj, grad!, rand(Float64, m*k + k*n), name = "NNMF")
end
