n, dens = 100, 0.1
model = qp_rand_model(n; dens = dens, convex = false)
@test all(-2.0 .≤ model.meta.lvar .≤ 0.0)
@test all(0.0 .≤ model.meta.uvar .≤ 2.0)
@test all(model.meta.x0 .== 0)

model = qp_rand_model(n; dens = dens, convex = true)
@test all(-2.0 .≤ model.meta.lvar .≤ 0.0)
@test all(0.0 .≤ model.meta.uvar .≤ 2.0)
@test all(model.meta.x0 .== 0)
