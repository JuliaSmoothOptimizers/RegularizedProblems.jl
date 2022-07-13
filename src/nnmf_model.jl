export nnmf_model

function nnmf_data(m::Int, n::Int, k::Int, T::DataType = Float64)

  params = [(zeros(Float64,n), zeros(Float64,(n,n))) for i in 1:k] 

  # generate mean vectors of the k clusters
  d = rand(1:k,n)
  v = convert(Vector{Float64},rand(1:3,n))
  for i in 1:n
    params[d[i]][1][i] = v[i]
  end

  # generate correlation matrices
  for i in 1:k
    for j in 1:n
      params[i][2][j,j] = params[i][1][j] > 0.0 ? 0.3 : 1e-8 # 1e-8 instead of 0.0 to avoid problems in Cholesky factorization when calling MixtureModel
    end
  end

  # generate a mixture of gaussians 
  dist = MixtureModel(MvNormal, params)

  # sample data 
  A = rand(dist, m)
  A[A .< 0] .= 0.0

  A
end

function nnmf_model(m::Int, n::Int, k::Int)
  A = nnmf_data(m, n, k)
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
    minusR = -reshape_array(r, (m,n)) # alloue
    W_T = reshape_array(x[1:(m*k)], (m,k))' # alloue
    H_T = reshape_array(x[(m*k+1):end], (k,n))' # alloue
    mul!(gw, minusR, H_T)
    mul!(gh, W_T, minusR)
    for i ∈ eachindex(g)
      g[i] = gw[i] - gh[i]
    end
    g
  end

  FirstOrderModel(obj, grad!, rand(Float64, m*k + k*n), name = "NNMF")
end
