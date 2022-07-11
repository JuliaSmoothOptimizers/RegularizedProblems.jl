export mat_rand_model

function mat_rand(m::Int, n::Int, r::Int, sr::Float64, va::Float64, vb::Float64, c::Float64)
  xl = Array(rand(Uniform(-0.3, 0.1), m, r))
  xr = Array(rand(Uniform(-0.3, 0.1), n, r))
  xs = xl * xr'
  Ω = findall(<(sr), rand(m, n))
  X = zeros(size(xs))
  X[Ω] = xs[Ω]
  B = xs[Ω]
  B = (1 - c) * add_gauss(B, va, 0) + c * add_gauss(B, vb, 0)
  return xs, B, Ω
end

function mat_rand_model(m::Int, n::Int, r::Int, sr::Float64, va::Float64, vb::Float64, c::Float64)
  T = mat_rand(m, n, r, sr, va, vb, c)
  res = zeros(m, n)

  function resid!(res, x)
    res[T[3]] .= T[2] - reshape_array(x, (m, n))[T[3]]
    vec(res)
  end

  function obj(x)
    resid!(res, x)
    dot(res, res) / 2
  end

  function grad!(g, x)
    resid!(res, x)
    res
  end
  function REL(x)
    rel = sqrt(norm(x - reshape_array(T[1], (m * n, 1))) / (m * n))
    rel
  end

  return FirstOrderModel(obj, grad!, rand(Float64, m * n), name = "MATRAND"), REL, T[1]
end
