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

  pars_FH = [0.5, 0.08, 1.0, 0.8, 0.7]
  prob_FH = DifferentialEquations.ODEProblem(FH_ODE, u0, tspan, pars_FH)

  x0 = [0, 0.2, 1.0, 0, 0]
  prob_VDP = DifferentialEquations.ODEProblem(FH_ODE, u0, tspan, x0)
  sol_VDP = DifferentialEquations.solve(prob_VDP, reltol = 1e-6, saveat = savetime)

  # add random noise to vdP solution
  t = sol_VDP.t
  b = vec(sol_VDP)
  noise = 0.1 * randn(size(b))
  data = noise .+ b

  # solve FH with parameters p
  function simulate(p, abstol = 1e-14, reltol = 1e-14)
    temp_prob = DifferentialEquations.remake(prob_FH, p = p)
    sol = DifferentialEquations.solve(
      temp_prob,
      DifferentialEquations.Vern9(),
      abstol = abstol,
      reltol = reltol,
      saveat = savetime,
    )
    # if any((sol.retcode != :Success for s in sol))
    #   @warn "ODE solution failed with parameters" p'
    #   error("ODE solution failed")
    # end
    return vec(sol)
  end

  # define residual vector
  function residual(p, args...)
    F = simulate(p, args...)
    F .-= data
    return F
  end

  # misfit = ‖residual‖² / 2
  function misfit(p, args...)
    F = residual(p, args...)
    return dot(F, F) / 2
  end

  return data, simulate, residual, misfit, x0
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
constructure, e.g., to set the automatic differentiation backend.

## Return Value

An instance of an `ADNLPModel` that represents the Fitzhugh-Nagumo problem, an instance
of an `ADNLSModel` that represents the same problem, and the exact solution.
"""
function fh_model(; kwargs...)
  data, simulate, resid, misfit, x0 = FH_smooth_term()
  nequ = 202
  ADNLPModels.ADNLPModel(misfit, ones(5); matrix_free = true, kwargs...),
  ADNLPModels.ADNLSModel(resid, ones(5), nequ; matrix_free = true, kwargs...),
  x0
end
