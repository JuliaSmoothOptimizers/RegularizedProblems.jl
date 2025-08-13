export group_lasso_model

function group_lasso_data(;
  m::Int = 200,
  n::Int = 512,
  g::Int = 16,
  ag::Int = 5,
  noise::Float64 = 0.01,
  compound::Int = 1,
)
  m ≤ n || error("number of rows ($m) should be ≤ number of columns ($n)")
  mod(n, g) == 0 || error("number of groups ($g) must divide evenly into number of rows ($n)")
  ag ≤ g || error("number of active groups ($ag) must be smaller than the number of groups ($g)")
  compound > 0 || error("compound factor must be positive")

  m = compound * m
  n = compound * n
  g = compound * g
  ag = compound * ag
  x0 = zeros(n)
  active_groups = sort(randperm(g)[1:ag]) # pick out active groups
  group_eles = Int(n / g) # get number of elements in a group
  xg = zeros(group_eles)
  indset = zeros(Int, g, group_eles)
  for i = 1:g
    if sum(i .== active_groups) > 0
      xg = sign(randn()) .* ones(group_eles)
      ind = Array(1:group_eles) .+ (group_eles * (i - 1)) # get index for active group
      x0[ind] = xg # put sparse signal in the main matrix
    end
    indset[i, :] = Array(1:group_eles) .+ (group_eles * (i - 1))
  end
  Q, _ = qr(randn(n, m))
  A = Array(Array(Q)')
  b0 = A * x0
  b = b0 + noise * randn(m)
  A, b, b0, x0, g, active_groups, indset
end

"""
    model, nls_model, sol = group_lasso_model(; kwargs...)

Return an instance of an `NLPModel` and `NLSModel` representing the group-lasso
problem, i.e., the under-determined linear least-squares objective

   ½ ‖Ax - b‖₂²,

where A has orthonormal rows and b = A * x̄ + ϵ, x̄ is sparse and ϵ is a noise
vector following a normal distribution with mean zero and standard deviation σ.
Note that with this format, all groups have a the same number of elements and the number of
groups divides evenly into the total number of elements.

## Keyword Arguments

* `m :: Int`: the number of rows of A (default: 200)
* `n :: Int`: the number of columns of A, with `n` ≥ `m` (default: 512)
* `g :: Int`: the number of groups (default: 16)
* `ag :: Int`: the number of active groups (default: 5)
* `noise :: Float64`: noise amount (default: 0.01)
* `compound :: Int`: multiplier for `m`, `n`, `g`, and `ag` (default: 1).

## Return Value

An instance of an `NLPModel` that represents the group-lasso problem.
An instance of an `NLSModel` that represents the group-lasso problem.
Also returns true x, number of groups g, group-index denoting which groups are active, and a Matrix where rows are group indices of x.
"""
function group_lasso_model(args...; kwargs...)
  A, b, b0, x0, g, active_groups, indset = group_lasso_data(args...; kwargs...)
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

  FirstOrderModel(obj, grad!, zero(x0), name = "Group Lasso"),
  FirstOrderNLSModel(
    resid!,
    jprod_resid!,
    jtprod_resid!,
    size(A, 1),
    zero(x0),
    name = "Group-Lasso-LS",
  ),
  x0,
  g,
  active_groups,
  indset
end
