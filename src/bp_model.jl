export bp_data, bp_model

function bp_data(m::Int, n::Int, k::Int; bounds::Bool = false)
  m ≤ n || error("number of rows ($m) should be ≤ number of columns ($n)")
  x0 = zeros(n)
  p = randperm(n)[1:k]
  # create sparse signal
  if bounds
    x0[p[1:k]] .= 1
  else
    x0[p[1:k]] = sign.(randn(k))
  end
  Q, _ = qr(randn(n, m))
  A = Array(Array(Q)')
  b = A * x0
  A, b, x0
end

bp_data(; compound::Int = 1, bounds::Bool = false) =
  bp_data(200 * compound, 512 * compound, 10 * compound, bounds = bounds)

"""
    model, sol = bp_model(A, b, x0; bounds = false)
    model, sol = bp_model(m, n, k; bounds = false)
    model, sol = bp_model(; compound = 1, bounds = false)

Return an instance of an `NLPModel` representing the basis-pursuit problem, 
i.e., the under-determined linear constraint

   Ax = b,

where A has orthonormal rows and b = A * x0, where x0 is sparse.

## Arguments

* `A :: AbstractMatrix{T}`: the matrix A, should have orthonormal rows;
* `b::AbstractVector{T}`: the right-hand side b;
* `x0::AbstractVector{T}`: the sparse signal x0.

The second form generates a matrix A of size m times n, a right hand side b and a sparse signal x0 
and calls the first calling form (see `bp_data`).

The third form calls the second form with arguments

    m = 200 * compound
    n = 512 * compound
    k =  10 * compound

## Keyword arguments

* `bounds :: Bool`: whether or not to include nonnegativity bounds in the model (default: false).

## Return Value

An instance of an `NLPModel` representing the
basis-pursuit problem, and the exact solution x̄.

If `bounds == true`, the positive part of x̄ is returned.
"""
bp_model(;compound::Int = 1, bounds::Bool = false) = 
  bp_model(200 * compound, 512 * compound, 10 * compound, bounds = bounds)

function bp_model(m::Int, n::Int, k::Int; bounds::Bool = false)
  A, b, x0 = bp_data(m, n, k; bounds = bounds)
  return bp_model(A, b, x0; bounds = bounds)
end

function bp_model(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T}; bounds::Bool = false) where{T}
  m, n = size(A)

  # Constrained API
  function cons!(cx, x)
    cx .= b
    mul!(cx, A, x, one(eltype(x)), -one(eltype(x)))
  end

  function jprod!(jv, x, v)
    mul!(jv, A, v)
  end

  function jtprod!(jtv, x, v)
    mul!(jtv, A', v)
  end

  function hprod!(hv, x, y, v; obj_weight = one(T))
    return hv .= zero(T)
  end

  function hess_coord!(vals, x, y; obj_weight = one(T))
    return zeros(T, 0)
  end

  rows_jac = repeat(1:m, outer = n)
  cols_jac = repeat(1:n, inner = m)

  function jac_coord!(vals, x)
    m, n = size(A)
    @inbounds for j in 1:n
      offset = (j-1)*m
      @inbounds for i in 1:m
        k = i + offset
        vals[k] = A[i, j]
      end
    end
  end

  # Unconstrained API
  function obj(x)
    return zero(T)
  end

  function grad!(g, x)
    g .= zero(T)
  end

  function hprod!(hv, x, v; obj_weight = one(T))
    return hv .= zero(T)
  end

  function hess_coord!(vals, x; obj_weight = one(T))
    return zeros(T, 0)
  end

  nlp = NLPModel(
    zero(x0), 
    obj, 
    lvar = (bounds ? zero(x0) : -Inf*ones(T, length(x0))),
    grad = grad!,
    hprod = hprod!,
    hess_coord = (zeros(T, 0), zeros(T, 0), hess_coord!),
    cons = (cons!, zero(b), zero(b)),
    jprod = jprod!,
    jtprod = jtprod!,
    jac_coord = (rows_jac, cols_jac, jac_coord!)
  )

  return nlp, x0
end
