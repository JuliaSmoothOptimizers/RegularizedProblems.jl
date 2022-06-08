export nnmf_model

function nnmf_data(m::Int, n::Int)
  A = Array(randn(n, m))
  A
end

function nnmf_model(m::Int, n::Int, k::Int)
  A = nnmf_data(m, n)
  r = vec(similar(A))
  WH = similar(A)
  gw = zeros((m,k))
  gh = zeros((k,n))

  function resid!(r, x)
    W = reshape(x[1:(m*k)], (m,k))
    H = reshape(x[(m*k+1):end], (k,n))
    mul!(WH, W, H)
    r .= vec(A - WH)
    r
  end

  function obj(x)
    resid!(r, x)
    dot(r, r) / 2
  end

  function grad!(g, x)
    resid!(r, x)
    R = reshape(r, (m,n))
    W = reshape(x[1:(m*k)], (m,k))
    H = reshape(x[(m*k+1):end], (k,n))
    mul!(gw, -R, Array(H'))
    mul!(gh, -Array(W'), R)
    g .= vcat(vec(gw), vec(gh))
    g
  end

  FirstOrderModel(obj, grad!, randn(m*k + k*n), name = "NNMF")
end
