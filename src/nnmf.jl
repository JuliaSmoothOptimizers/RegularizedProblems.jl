export nnmf_model

function nnmf_data(m::Int, n::Int, k::Int, T::DataType = Float64)
  parameters = [(zeros(T, n), zeros(T, (n, n))) for i = 1:k]
  # generate mean vectors of the k clusters
  d = rand(1:k, n)
  v = convert(Vector{T}, rand(1:3, n))
  for i = 1:n
    parameters[d[i]][1][i] = v[i]
  end
  # generate correlation matrices
  ϵ = sqrt(eps(T))
  for i = 1:k
    for j = 1:n
      parameters[i][2][j, j] = parameters[i][1][j] > 0.0 ? 0.3 : ϵ # to avoid problems in Cholesky factorization when calling MixtureModel
    end
  end
  # generate a mixture of gaussians 
  dist = MixtureModel(MvNormal, parameters)
  # sample data 
  A = rand(dist, m)'
  A[A .< 0] .= 0
  return A
end

function nnmf_model(m::Int = 100, n::Int = 50, k::Int = 10, T::DataType = Float64)
  A = nnmf_data(m, n, k, T)
  r = similar(A, m * n)
  WH = similar(A)
  gw = similar(A, (m, k))
  gh = similar(A, (k, n))
  selected = (m * k + 1):((m + n) * k)

  function resid!(r, x)
    W = reshape_array(view(x, 1:(m * k)), (m, k))
    H = reshape_array(view(x, (m * k + 1):((m + n) * k)), (k, n))
    mul!(WH, W, H)
    for i ∈ eachindex(r)
      r[i] = A[i] - WH[i]
    end
    return r
  end

  function obj(x)
    resid!(r, x)
    return dot(r, r) / 2
  end

  function grad!(g, x)
    resid!(r, x)
    minusR = reshape_array(r, (m, n))
    minusR .*= -1
    W_T = reshape_array(view(x, 1:(m * k)), (m, k))'
    H_T = reshape_array(view(x, (m * k + 1):((m + n) * k)), (k, n))'
    mul!(gw, minusR, H_T)
    mul!(gh, W_T, minusR)
    for i ∈ eachindex(gw)
      g[i] = gw[i]
    end
    for i ∈ eachindex(gh)
      g[i + m * k] = gh[i]
    end
    return g
  end

  function jacv!(Jv, x, v)
    W = reshape_array(view(x, 1:(m * k)), (m, k))
    H = reshape_array(view(x, (m * k + 1):((m + n) * k)), (k, n))
    W_v = reshape_array(view(v, 1:(m * k)), (m, k))
    H_v = reshape_array(view(v, (m * k + 1):((m + n) * k)), (k, n))
    mul!(WH, W_v, H)
    for i ∈ eachindex(WH)
      Jv[i] = -WH[i]
    end
    mul!(WH, W, H_v)
    for i ∈ eachindex(WH)
      Jv[i] -= WH[i]
    end
    return Jv
  end

  function jactv!(Jtv, x, w)
    W_T = reshape_array(view(x, 1:(m * k)), (m, k))'
    H_T = reshape_array(view(x, (m * k + 1):((m + n) * k)), (k, n))'
    X_v = reshape_array(w, (m, n))
    mul!(gw, X_v, H_T)
    mul!(gh, W_T, X_v)
    for i ∈ eachindex(gw)
      Jtv[i] = -gw[i]
    end
    for i ∈ eachindex(gh)
      Jtv[i + m * k] = -gh[i]
    end
    return Jtv
  end
  x0 = 3 * rand(eltype(A), k * (m + n))

  FirstOrderModel(
    obj,
    grad!,
    x0,
    name = "NNMF",
    lvar = zeros(eltype(A), k * (m + n)),
    uvar = fill!(zeros(eltype(A), k * (m + n)), Inf),
  ),
  FirstOrderNLSModel(
    resid!,
    jacv!,
    jactv!,
    m * n,
    x0,
    name = "NNMF-LS",
    lvar = zeros(eltype(A), k * (m + n)),
    uvar = fill!(zeros(eltype(A), k * (m + n)), Inf),
  ),
  A[:],
  selected
end
