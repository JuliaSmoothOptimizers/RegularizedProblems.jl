export mat_rand_model

function mat_rand(m::Int, n::Int, r::Int, sr::Float64, va::Float64, vb::Float64, c::Float64)
  xl = rand(Uniform(-0.1, 0.3), m, r)
  xr = rand(Uniform(-0.1, 0.3), n, r)
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
    res .= zeros(m, n)
    res[Ω] .= reshape_array(x, (m, n))[Ω] .- B
    vec(res)
  end

  function obj(x)
    resid!(res, x)
    dot(res, res) / 2
  end

  grad!(r, x) = resid!(reshape_array(r, (m, n)), x)

  return FirstOrderModel(obj, grad!, rand(Float64, m * n), name = "MATRAND"), xs, B
end
