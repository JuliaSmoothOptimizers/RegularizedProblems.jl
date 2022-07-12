export mat_rand_model

function mat_rand(m::Int, n::Int, r::Int, sr::Float64, va::Float64, vb::Float64, c::Float64)
  xl = Array(rand(Uniform(-0.3, 0.1), m, r))
  xr = Array(rand(Uniform(-0.3, 0.1), n, r))
  xs = xl * xr'
  Ω = findall(<(sr), rand(m, n))
  B = xs[Ω]
  B = (1 - c) * add_gauss(B, va, 0) + c * add_gauss(B, vb, 0)
  return xs, B, Ω
end

function mat_rand_model(m::Int, n::Int, r::Int, sr::Float64, va::Float64, vb::Float64, c::Float64)
  xs, B, Ω = mat_rand(m, n, r, sr, va, vb, c)
  res = zeros(m, n)

  function resid!(res, x)
    res[Ω] .= B .- reshape_array(x, (m, n))[Ω]
    vec(res)
  end

  function obj(x)
    resid!(res, x)
    dot(res, res) / 2
  end

  function grad!(x, g)
    resid!(res, x)
    g .= res
    g
  end

  function REL(x)
    rel = sqrt(norm(x - reshape_array(xs, (m * n, 1))) / (m * n))
    rel
  end

  return FirstOrderModel(obj, grad!, rand(Float64, m * n), name = "MATRAND"), REL, xs
end
