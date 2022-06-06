export group_lasso_model, group_lasso_nls_model

function group_lasso_data(m::Int, n::Int, g::Int, ag::Int, noise::Float64 = 0.01)

  (m ≤ n) || error("number of rows ($m) should be ≤ number of columns ($n)")
  (mod(n, g) == 0) || error("number of groups ($g) must divide evenly into number of rows ($n)")
  (ag ≤ g) || error("number of active groups ($ag) must be smaller than the number of groups ($g)")
  # (k ≤ Int(n/g)) || error("number of sparse points ($k) must be smaller than the number of points in each group ($(n/g))")

  x0 = zeros(n)
  active_groups = sort(randperm(g)[1:ag]) # pick out active groups
  group_eles = Int(n/g) # get number of elements in a group
  xg = zeros(group_eles)
  indset = zeros(Int, g, group_eles)
  for i = 1:g
    if sum(i .== active_groups) > 0
      # k = rand(big.(1:group_eles)) # generate number of sparse points in active group
      # p = randperm(group_eles)[1:k] # get list of those sparse points
      # xg[p] = sign.(randn(k)) # create sparse signal
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
    model, sol = bpdn_model(args...)
    model, sol = bpdn_model(compound = 1, args...)

Return an instance of an `NLPModel` representing the basis-pursuit denoise
problem, i.e., the under-determined linear least-squares objective

   ½ ‖Ax - b‖₂²,

where A has orthonormal rows and b = A * x̄ + ϵ, x̄ is sparse and ϵ is a noise
vector following a normal distribution with mean zero and standard deviation σ.

## Arguments

* `m :: Int`: the number of rows of A
* `n :: Int`: the number of columns of A (with `n` ≥ `m`)
* `k :: Int`: the number of nonzero elements in x̄
* `noise :: Float64`: noise amount ϵ (default: 0.01).

The second form calls the first form with arguments

    m = 200 * compound
    n = 512 * compound
    k =  10 * compound

## Return Value

An instance of a `FirstOrderModel` that represents the basis-pursuit denoise problem
and the exact solution x̄.
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

  FirstOrderModel(obj, grad!, zero(x0), name = "Group Lasso"), x0, g, active_groups, indset
end

"""
    model, sol = group_lasso_nls_model(args...)
    model, sol = group_lasso_nls_model(compound = 1, args...)

Return an instance of a `FirstOrderNLSModel` that represents the basis-pursuit
denoise problem explicitly as a least-squares problem and the exact solution x̄.

See the documentation of `group_lasso_model()` for more information and a
description of the arguments.
"""
function group_lasso_nls_model(args...)
  A, b, b0, x0, g, active_groups, indset = group_lasso_data(args...)
  r = similar(b)

  function resid!(r, x)
    mul!(r, A, x)
    r .-= b
    r
  end

  jprod_resid!(Jv, x, v) = mul!(Jv, A, v)
  jtprod_resid!(Jtv, x, v) = mul!(Jtv, A', v)

  FirstOrderNLSModel(resid!, jprod_resid!, jtprod_resid!, size(A, 1), zero(x0), name = "Group-Lasso-LS"), x0, g, active_groups, indset
end
