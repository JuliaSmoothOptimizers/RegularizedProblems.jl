export bpdn_model, bpdn_nls_model

function bpdn_data(m::Int, n::Int, k::Int, noise::Float64 = 0.01)
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

bpdn_data(compound::Int = 1, args...) =
    bpdn_data(200 * compound, 512 * compound, 10 * compound, args...)

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
function bpdn_model(args...)
    A, b, b0, x0 = bpdn_data(args...)
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

    FirstOrderModel(obj, grad!, zero(x0), name = "BPDN"), x0
end

"""
    model, sol = bpdn_nls_model(args...)
    model, sol = bpdn_nls_model(compound = 1, args...)

Return an instance of a `FirstOrderNLSModel` that represents the basis-pursuit
denoise problem explicitly as a least-squares problem and the exact solution x̄.

See the documentation of `bpdn_model()` for more information and a
description of the arguments.
"""
function bpdn_nls_model(args...)
    A, b, b0, x0 = bpdn_data(args...)
    r = similar(b)

    function resid!(r, x)
        mul!(r, A, x)
        r .-= b
        r
    end

    jprod_resid!(Jv, x, v) = mul!(Jv, A, v)
    jtprod_resid!(Jtv, x, v) = mul!(Jtv, A', v)

    FirstOrderNLSModel(
        resid!,
        jprod_resid!,
        jtprod_resid!,
        size(A, 1),
        zero(x0),
        name = "BPDN-LS",
    ),
    x0
end
