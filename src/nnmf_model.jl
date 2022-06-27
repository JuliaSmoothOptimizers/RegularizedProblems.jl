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

  reshape2(a, dims) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)

  function resid!(r, x)
    W = reshape2(x[1:(m*k)], (m,k))
    H = reshape2(x[(m*k+1):end], (k,n))
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
    R = reshape2(r, (m,n))
    W = reshape2(x[1:(m*k)], (m,k))
    H = reshape2(x[(m*k+1):end], (k,n))
    mul!(gw, -R, Array(H'))
    mul!(gh, -Array(W'), R)
    g .= vcat(vec(gw), vec(gh))
    g
  end

  FirstOrderModel(obj, grad!, rand(Float64, m*k + k*n), name = "NNMF")
end
