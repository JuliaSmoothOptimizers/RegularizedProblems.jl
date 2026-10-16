export fh_model, fh_nls_model

# Right-hand sides.  p = (I, μ, a, b, c)
# FH model = van der Pol oscillator when I = b = c = 0.
function FH_ODE!(dx, x, p, t)
  V, W = x[1], x[2]
  I, μ, a, b, c = p
  dx[1] = (V - V^3 / 3 - W + I) / μ
  dx[2] = μ * (a * V - b * W + c)
end

# Forward sensitivity system, x = [V, W, ∂V/∂p₁, ∂W/∂p₁, …, ∂V/∂p₅, ∂W/∂p₅]:
#   u' = f(u, p),   S' = (∂f/∂u) S + ∂f/∂p,   S(0) = 0.
# One solve of this 12-state system gives F(x) and the full 202×5 Jacobian.
function FH_sens!(dz, z, p, t)
  V, W = z[1], z[2]
  I, μ, a, b, c = p
  FH_ODE!(view(dz, 1:2), view(z, 1:2), p, t)  # compute the first two entries of dz
  # ∂f/∂u
  j11, j12 = (1 - V^2) / μ, -1 / μ
  j21, j22   = μ * a, -μ * b
  # ∂f/∂p, one entry per parameter (I, μ, a, b, c)
  q1 = (1 / μ, -dz[1] / μ, zero(V), zero(V), zero(V))
  q2 = (zero(V), dz[2] / μ, μ * V, -μ * W, μ)
  for k in 1:5
    dz[2k + 1] = j11 * z[2k + 1] + j12 * z[2k + 2] + q1[k]
    dz[2k + 2] = j21 * z[2k + 1] + j22 * z[2k + 2] + q2[k]
  end
end

# ---------------------------------------------------------------------------
# Shared cache: integrators, data, and the last F / J keyed on x.
# ---------------------------------------------------------------------------
mutable struct FHCache{T, I2, I12}
  integ::I2           # 2-state integrator (residual only)
  sinteg::I12         # 12-state integrator (residual + Jacobian)
  u0::Vector{T}
  z0::Vector{T}
  ts::Vector{T}       # observation times
  data::Matrix{T}     # nt × 2
  buf2::Vector{T}
  buf12::Vector{T}
  F::Vector{T}        # 2nt,  F[k] ↔ V(t_k),  F[nt + k] ↔ W(t_k)
  J::Matrix{T}        # 2nt × 5
  xF::Vector{T}       # last x for which F is valid
  xJ::Vector{T}       # last x for which J is valid
  F_valid::Bool
  J_valid::Bool
  F_success::Bool
  J_success::Bool
end

@inline function _record!(c::FHCache, k, u, ::Val{false})
  nt = length(c.ts)
  @inbounds begin
    c.F[k] = u[1] - c.data[k, 1]
    c.F[k + nt] = u[2] - c.data[k, 2]
  end
end

@inline function _record!(c::FHCache, k, z, ::Val{true})
  _record!(c, k, z, Val(false))
  nt = length(c.ts)
  @inbounds for j in 1:5
    c.J[k, j] = z[2j + 1]
    c.J[k + nt, j] = z[2j + 2]
  end
end

# Integrate with parameters x and record the solution at every c.ts[k].
function _simulate!(c::FHCache, integ, uinit, buf, x, sens::Val)
  copyto!(integ.p, x)
  OrdinaryDiffEqVerner.reinit!(integ, uinit)
  ts = c.ts
  nt = length(ts)
  copyto!(buf, uinit)
  _record!(c, 1, buf, sens)
  k = 2
  while k <= nt
    OrdinaryDiffEqVerner.step!(integ)
    OrdinaryDiffEqVerner.check_error(integ) == OrdinaryDiffEqVerner.ReturnCode.Success || return false
    while k <= nt && ts[k] <= integ.t
      integ(buf, ts[k])                     # in-place interpolation, no allocation
      _record!(c, k, buf, sens)
      k += 1
    end
  end
  return true
end

function _update_F!(c::FHCache, x)
  (c.F_valid && x == c.xF) && return c.F_success
  sim_success = _simulate!(c, c.integ, c.u0, c.buf2, x, Val(false))
  sim_success || fill!(c.F, NaN)
  copyto!(c.xF, x)
  c.F_valid, c.F_success = true, sim_success
  return c.F_success
end

function _update_J!(c::FHCache, x)
  (c.J_valid && x == c.xJ) && return c.J_success
  sim_success = _simulate!(c, c.sinteg, c.z0, c.buf12, x, Val(true))   # also refreshes F
  if !sim_success
    fill!(c.F, NaN)
    fill!(c.J, NaN)
  end
  copyto!(c.xJ, x)
  copyto!(c.xF, x)
  c.J_valid, c.F_valid = true, true
  c.J_success, c.F_success = sim_success, sim_success
  return c.J_success
end

function _fh_cache(
  xtrue::Vector{Float64};
  rng::Random.AbstractRNG = Random.default_rng(),
  noise_level = 0.1,
  abstol = 1e-14,
  reltol = 1e-14,
  u0 = [2.0, 0.0],
  tf = 20.0,
  savetime = 0.2,
)
  T = Float64
  ts = collect(T, range(zero(T), tf; step = savetime))
  tspan = (ts[1], ts[end])
  nt = length(ts)
  alg = OrdinaryDiffEqVerner.Vern9()
  opts = (; abstol, reltol, save_everystep = false, save_start = false, save_end = false)

  prob2 = OrdinaryDiffEqVerner.ODEProblem(FH_ODE!, copy(u0), tspan, copy(xtrue))
  integ = OrdinaryDiffEqVerner.init(prob2, alg; opts...)
  z0 = [u0; zeros(T, 10)]
  prob12 = OrdinaryDiffEqVerner.ODEProblem(FH_sens!, copy(z0), tspan, copy(xtrue))
  sinteg = OrdinaryDiffEqVerner.init(prob12, alg; opts...)

  c = FHCache(
    integ, sinteg, copy(u0), z0, ts, zeros(T, nt, 2), zeros(T, 2), zeros(T, 12),
    zeros(T, 2nt), zeros(T, 2nt, 5), fill(T(NaN), 5), fill(T(NaN), 5),
    false, false, false, false,
  )

  # Synthetic data: with data = 0, F = [V(t); W(t)] at the true parameters.
  _simulate!(c, integ, c.u0, c.buf2, xtrue, Val(false)) ||
    error("FitzHugh-Nagumo: integration failed at the true parameters")
  c.data .= reshape(c.F, nt, 2) .+ noise_level .* permutedims(reshape(randn(rng, 2nt), 2, nt))
  c.F_valid = false   # F currently holds the clean trajectory, not a residual
  return c
end

# ---------------------------------------------------------------------------
# NLPModels API
# ---------------------------------------------------------------------------

mutable struct FHNLPModel{T, S, C} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  cache::C
end

mutable struct FHNLSModel{T, S, C} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
  cache::C
end

for M in (:FHNLPModel, :FHNLSModel)
  @eval begin
    function NLPModels.obj(m::$M{T}, x::AbstractVector) where {T}
      NLPModels.@lencheck m.meta.nvar x
      NLPModels.increment!(m, :neval_obj)
      c = m.cache
      _update_F!(c, x) || return T(Inf)
      return dot(c.F, c.F) / 2
    end

    function NLPModels.grad!(m::$M, x::AbstractVector, g::AbstractVector)
      NLPModels.@lencheck m.meta.nvar x g
      NLPModels.increment!(m, :neval_grad)
      c = m.cache
      _update_J!(c, x)
      _update_F!(c, x)              # no-op unless F was overwritten at another x
      mul!(g, c.J', c.F)            # ∇f = Jᵀ F
      return g
    end
  end
end

function NLPModels.residual!(nls::FHNLSModel, x::AbstractVector, Fx::AbstractVector)
  NLPModels.@lencheck nls.meta.nvar x
  NLPModels.@lencheck nls.nls_meta.nequ Fx
  NLPModels.increment!(nls, :neval_residual)
  _update_F!(nls.cache, x)
  copyto!(Fx, nls.cache.F)
  return Fx
end

function NLPModels.jprod_residual!(
  nls::FHNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  NLPModels.@lencheck nls.meta.nvar x v
  NLPModels.@lencheck nls.nls_meta.nequ Jv
  NLPModels.increment!(nls, :neval_jprod_residual)
  _update_J!(nls.cache, x)
  mul!(Jv, nls.cache.J, v)
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::FHNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  NLPModels.@lencheck nls.meta.nvar x Jtv
  NLPModels.@lencheck nls.nls_meta.nequ v
  NLPModels.increment!(nls, :neval_jtprod_residual)
  _update_J!(nls.cache, x)
  mul!(Jtv, nls.cache.J', v)
  return Jtv
end

"""
    nlp, nls, xtrue = fh_model(; x0 = ones(5), rng = Random.default_rng(), kwargs...)

Return an `NLPModel` and an `NLSModel` representing the same Fitzhugh-Nagumo problem,
i.e., the over-determined nonlinear least-squares objective

   ½ ‖F(x)‖₂²,

where F: ℝ⁵ → ℝ²⁰² is the fitting error between a simulation of the Fitzhugh-Nagumo
model with parameters x and noisy observations of a Van der Pol oscillator with fixed,
but unknown, parameters `xtrue`.

Derivatives are exact (up to integration tolerance): F and its Jacobian are obtained in
a single solve of the forward-sensitivity system and cached for the last `x`, so repeated
`grad!`, `jprod_residual!` and `jtprod_residual!` calls at the same `x` cost one 202×5
matrix-vector product. Both models share the cache (not thread-safe).

Keyword arguments other than `x0` and `name` (`rng`, `noise_level`, `abstol`, `reltol`,
`u0`, `tf`, `savetime`) configure the simulation.
"""
function fh_model(; x0 = ones(5), name = "Fitzhugh-Nagumo", kwargs...)
  xtrue = [0.0, 0.2, 1.0, 0.0, 0.0]
  c = _fh_cache(xtrue; kwargs...)
  nequ = length(c.F)
  nlp = FHNLPModel(NLPModelMeta(5; x0 = copy(x0), name = name), Counters(), c)
  nls = FHNLSModel(
    NLPModelMeta(5; x0 = copy(x0), name = name * "-LS"),
    NLSMeta(nequ, 5; x0 = copy(x0)),
    NLSCounters(),
    c,
  )
  return nlp, nls, xtrue
end