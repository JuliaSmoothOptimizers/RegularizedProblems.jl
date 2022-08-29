export bpdn_constr_model

function bpdn_constr_data(m::Int, n::Int, k::Int, noise::Float64 = 0.01)
  m ≤ n || error("number of rows ($m) should be ≤ number of columns ($n)")
  x0 = zeros(n)
  p = randperm(n)[1:k]
  x0[p[1:k]] = sign.(randn(k)) # create sparse signal
  Q, _ = qr(randn(n, m))
  A = Array(Array(Q)')
  b0 = A * x0
  b = b0 + noise * randn(m)
  A, b, b0, x0
end

bpdn_constr_data(compound::Int = 1, args...) =
  bpdn_constr_data(200 * compound, 512 * compound, 10 * compound, args...)


function bpdn_constr_model(args...)
  A, b, b0, x0 = bpdn_constr_data(args...)
  r = similar(b)

  function resid!(r, x)
    mul!(r, A, x)
    r .-= b
    r
  end

  jprod_resid!(Jv, x, v) = mul!(Jv, A, v)
  jtprod_resid!(Jtv, x, v) = mul!(Jtv, A', v)

  function obj(x)
    resid!(r, x)
    dot(r, r) / 2
  end

  function grad!(g, x)
    resid!(r, x)
    mul!(g, A', r)
    g
  end

  FirstOrderModel(obj, grad!, zero(x0), name = "BPDN-constrained", lvar = zero(x0), uvar = fill!(Inf, similar(x0))),
  x0
end
