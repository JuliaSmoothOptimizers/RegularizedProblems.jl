using Test
using LinearAlgebra
using RegularizedProblems
using NLPModels
using Random

@testset "Binomial Model Correctness" begin
    # 1. Setup Synthetic Data
    Random.seed!(42)
    m, n = 10, 100  # 10 features, 100 samples (overdetermined for strict convexity)
    A = randn(m, n)
    
    # Generate labels based on a true model
    x_true = randn(m)
    logits = A' * x_true
    probs_true = 1.0 ./ (1.0 .+ exp.(-logits))
    b = Float64.(rand(n) .< probs_true)

    nlp = binomial_model(A, b)
    x = randn(m) # Random point for checking

    # 2. Gradient Verification
    g = grad(nlp, x)
    
    # Manual Gradient Calculation
    # f(x) = sum(log(1 + exp(a_i' x)) - y_i a_i' x)
    # ∇f(x) = A * (sigmoid(A'x) - y)
    z = A' * x
    p = 1.0 ./ (1.0 .+ exp.(-z))
    g_manual = A * (p - b)
    
    @test g ≈ g_manual atol=1e-10

    # 3. Hessian-Vector Product Verification
    v = randn(m)
    hv = hprod(nlp, x, v)
    
    # Manual Hessian-vector Calculation
    # ∇²f(x) = A * diag(p .* (1 - p)) * A'
    # H v = A * ( (p .* (1 - p)) .* (A' v) )
    w = p .* (1.0 .- p)
    hv_manual = A * (w .* (A' * v))
    
    @test hv ≈ hv_manual atol=1e-10

    # 4. Newton Method Convergence (Matrix-Free)
    # We implement a simple Newton-CG solver to verify the model works in optimization
    println("\nRunning Newton-CG on Binomial Model...")
    x_k = zeros(m)
    
    for iter in 1:20
        g_k = grad(nlp, x_k)
        gnorm = norm(g_k)
        # println("  Iter $iter: ||g|| = $gnorm")
        if gnorm < 1e-6
            break
        end
        
        # Solve H d = -g using Conjugate Gradient
        d = zeros(m)
        r = -g_k - hprod(nlp, x_k, d)
        p_cg = copy(r)
        rsold = dot(r, r)
        
        for cg_iter in 1:m
            Hp = hprod(nlp, x_k, p_cg)
            alpha = rsold / dot(p_cg, Hp)
            d .+= alpha .* p_cg
            r .-= alpha .* Hp
            rsnew = dot(r, r)
            if sqrt(rsnew) < 1e-10 break end
            p_cg .= r .+ (rsnew / rsold) .* p_cg
            rsold = rsnew
        end
        
        x_k .+= d
    end
    
    # Verify we reached the optimum
    g_final = grad(nlp, x_k)
    @test norm(g_final) < 1e-5
    println("Newton-CG converged with ||g|| = $(norm(g_final))")
end