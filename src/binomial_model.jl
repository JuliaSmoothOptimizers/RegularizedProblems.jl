export binomial_model

"""
    nlp_model, nls_model = binomial_model(A, b)

Return an instance of an `NLPModel` representing the binomial logistic regression
problem, and an `NLSModel` representing the system of equations formed by its gradient.

Minimize f(x) = sum(log(1+exp(a_i^T x)) - y_i a_i^T x)
where y_i in {0, 1}.

The NLS residual is defined as the gradient of f(x):
F(x) = A (p - b)
where p = sigmoid(A' x).
"""
function binomial_model(A, b)
  m, n = size(A) # m features, n samples
  
  # Pre-allocate buffers
  Ax = zeros(n)
  p = zeros(n)
  w = zeros(n)
  tmp_n = zeros(n)

  function resid!(r, x)
    mul!(Ax, A', x)
    @. p = 1.0 / (1.0 + exp(-Ax))
    @. tmp_n = p - b
    mul!(r, A, tmp_n)
    return r
  end

  function jacv!(Jv, x, v)
    mul!(Ax, A', x)
    @. w = 1.0 / (1.0 + exp(-Ax))
    @. w = w * (1.0 - w)
    
    mul!(tmp_n, A', v)
    @. tmp_n *= w
    mul!(Jv, A, tmp_n)
    return Jv
  end

  function jactv!(Jtv, x, v)
    jacv!(Jtv, x, v)
  end

  function obj(x)
    mul!(Ax, A', x)
    # Stable computation of log(1+exp(z)) - b*z
    return sum(@. (Ax > 0) * ((1 - b) * Ax + log(1 + exp(-Ax))) + (Ax <= 0) * (log(1 + exp(Ax)) - b * Ax))
  end

  function grad!(g, x)
    resid!(g, x)
  end

  x0 = zeros(m)
  
  return FirstOrderModel(obj, grad!, x0, name="Binomial"),
         FirstOrderNLSModel(resid!, jacv!, jactv!, m, x0)
end