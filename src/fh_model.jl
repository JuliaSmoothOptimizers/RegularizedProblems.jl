export fh_model, fh_nls_model

function FH_smooth_term(; abstol = 1e-14, reltol = 1e-14)
  # FH model = van-der-Pol oscillator when I = b = c = 0
  # x' = μ(x - x^3/3 - y)
  # y' = x/μ -> here μ = 12.5
  function FH_ODE(dx, x, p, t)
    V, W = x
    I, μ, a, b, c = p
    dx[1] = (V - V^3 / 3 - W + I) / μ
    dx[2] = μ * (a * V - b * W + c)
  end

  u0 = [2.0; 0.0]
  tspan = (0.0, 20.0)
  savetime = 0.2

  x0 = [0.0, 0.2, 1.0, 0.0, 0.0]
  p = copy(x0)
  prob = OrdinaryDiffEqVerner.ODEProblem(FH_ODE, u0, tspan, x0)
  alg = OrdinaryDiffEqVerner.Vern9()
  integrator = OrdinaryDiffEqVerner.init(prob, alg, abstol = abstol, reltol = reltol, p = p, saveat = savetime)
  OrdinaryDiffEqVerner.solve!(integrator)


  # add random noise to vdP solution
  noise = 0.1 * randn(length(integrator.sol.u), 2)
  noise = collect(eachrow(noise))
  data = noise + integrator.sol.u
  temp = zeros(eltype(data[1]), length(data) * 2)

  # define residual vector
  function residual!(F, x)
    OrdinaryDiffEqVerner.reinit!(integrator, u0)
    integrator.p .= x
    OrdinaryDiffEqVerner.solve!(integrator)
    @inbounds for i in 1:length(integrator.sol.u)
      F[i] = integrator.sol.u[i][1] - data[i][1]
      F[i + length(integrator.sol.u)] = integrator.sol.u[i][2] - data[i][2]
    end
    return F
  end

  function obj(x)
    residual!(temp, x)
    return dot(temp, temp) / 2
  end

  return data, residual!, obj, x0
end

"""
    fh_model(; kwargs...)

Return an instance of an `NLPModel` and an instance of an `NLSModel` representing
the same Fitzhugh-Nagumo problem, i.e., the over-determined nonlinear
least-squares objective

   ½ ‖F(x)‖₂²,

where F: ℝ⁵ → ℝ²⁰² represents the fitting error between a simulation of the
Fitzhugh-Nagumo model with parameters x and a simulation of the Van der Pol
oscillator with fixed, but unknown, parameters.

## Keyword Arguments

All keyword arguments are passed directly to the `ADNLPModel` (or `ADNLSModel`)
constructor.

## Return Value

An instance of an `ADNLPModel` that represents the Fitzhugh-Nagumo problem, an instance
of an `ADNLSModel` that represents the same problem, and the exact solution.
"""
function fh_model(; kwargs...)
  data, resid!, misfit, x0 = FH_smooth_term()
  nequ = 202
  nlp = ADNLPModels.ADNLPModel(misfit, ones(5); 
    backend = :enzyme,
    matrix_free = true,
    kwargs...
  )
  nls = ADNLPModels.ADNLSModel!(resid!, ones(5), nequ; 
    backend = :enzyme,
    matrix_free = true,
    kwargs...
  )
  return nlp, nls, x0
end