export fh_model, fh_nls_model

function FH_smooth_term()
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
  function simulate(p)
    temp_prob = DifferentialEquations.remake(prob_FH, p = p)
    sol = DifferentialEquations.solve(
      temp_prob,
      DifferentialEquations.Vern9(),
      abstol = 1e-14,
      reltol = 1e-14,
      saveat = savetime,
    )
    # if any((sol.retcode != :Success for s in sol))
    #   @warn "ODE solution failed with parameters" p'
    #   error("ODE solution failed")
    # end
    return vec(sol)
  end

  # define residual vector
  function residual(p)
    F = simulate(p)
    F .-= data
    return F
  end

  # misfit = ‖residual‖² / 2
  function misfit(p)
    F = residual(p)
    return dot(F, F) / 2
  end

  return data, simulate, residual, misfit, x0
end

function fh_model(; kwargs...)
  data, simulate, resid, misfit, x0 = FH_smooth_term()
  nequ = 202
  ADNLPModels.ADNLPModel(misfit, ones(5); kwargs...),
  ADNLPModels.ADNLSModel(resid, ones(5), nequ; kwargs...),
  x0
end
