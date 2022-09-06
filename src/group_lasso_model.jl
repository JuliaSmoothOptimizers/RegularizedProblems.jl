export group_lasso_model

function group_lasso_data(m::Int, n::Int, g::Int, ag::Int, noise::Float64 = 0.01)

  (m ≤ n) || error("number of rows ($m) should be ≤ number of columns ($n)")
  (mod(n, g) == 0) || error("number of groups ($g) must divide evenly into number of rows ($n)")
  (ag ≤ g) || error("number of active groups ($ag) must be smaller than the number of groups ($g)")

  x0 = zeros(n)
  active_groups = sort(randperm(g)[1:ag]) # pick out active groups
  group_eles = Int(n/g) # get number of elements in a group
  xg = zeros(group_eles)
  indset = zeros(Int, g, group_eles)
  for i = 1:g
    if sum(i .== active_groups) > 0
      xg = sign(randn()) .*ones(group_eles,)
      ind = Array(1:group_eles) .+ (group_eles * (i-1)) # get index for active group
      x0[ind] = xg # put sparse signal in the main matrix
    end
    indset[i,:] = Array(1:group_eles) .+ (group_eles * (i-1))
  end
  Q, _ = qr(randn(n, m))
  A = Array(Array(Q)')
  b0 = A * x0
  b = b0 + noise * randn(m)
  A, b, b0, x0, g, active_groups, indset
end

group_lasso_data(compound::Int = 1, args...) =
  group_lasso_data(200 * compound, 512 * compound, 16 * compound, 5 * compound, args...)

"""
    model, nls_model, sol = group_lasso_model(args...)
    model, nls_model, sol = group_lasso_model(compound = 1, args...)

Return an instance of an `NLPModel` and `NLSModel` representing the basis-pursuit denoise
problem, i.e., the under-determined linear least-squares objective

   ½ ‖Ax - b‖₂²,

where A has orthonormal rows and b = A * x̄ + ϵ, x̄ is sparse and ϵ is a noise
vector following a normal distribution with mean zero and standard deviation σ.

## Arguments

* `m :: Int`: the number of rows of A
* `n :: Int`: the number of columns of A (with `n` ≥ `m`)
* `g :: Int : the number of groups`
* `ag :: Int`: the number of active groups
* `noise :: Float64`: noise amount ϵ (default: 0.01).

The second form calls the first form with arguments

    m = 200 * compound
    n = 512 * compound
    k =  10 * compound

## Return Value

An instance of a `FirstOrderModel` that represents the basis-pursuit denoise problem
and the exact solution x̄.
An instance of a `FirstOrderNLSModel` that represents the basis-pursuit denoise problem
and the exact solution x̄.
Also returns true x, number of groups g, active groups (which ones in g), and active group indices (of x)
"""
function group_lasso_model(args...)
  A, b, b0, x0, g, active_groups, indset = group_lasso_data(args...)
  r = similar(b)

  function resid!(r, x)
    mul!(r, A, x)
    r .-= b
    r
  end

  function obj(x)
    resid!(r, x)
    dot(r, r) / 2
  end

  function grad!(g, x)
    resid!(r, x)
    mul!(g, A', r)
    g
  end

  jprod_resid!(Jv, x, v) = mul!(Jv, A, v)
  jtprod_resid!(Jtv, x, v) = mul!(Jtv, A', v)

  FirstOrderModel(obj, grad!, zero(x0), name = "Group Lasso"), FirstOrderNLSModel(resid!, jprod_resid!, jtprod_resid!, size(A, 1), zero(x0), name = "Group-Lasso-LS"), x0, g, active_groups, indset
end
