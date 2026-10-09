# Measure allocations inside a function, after a warm-up call.
function allocs_op(op!, y, x)
  op!(y, x)
  return @allocated op!(y, x)
end

function allocs_obj(model, x)
  obj(model, x)
  return @allocated obj(model, x)
end

function allocs_grad(model, x, g)
  grad!(model, x, g)
  return @allocated grad!(model, x, g)
end

function allocs_objgrad(model, x, g)
  objgrad!(model, x, g)
  return @allocated objgrad!(model, x, g)
end

# Straightforward (allocating) blur with complex FFTs, following the original implementation:
# zero-pad the image above and to the left, circularly convolve with the centered kernel, crop.
function reference_blur(x, kernel, shape, shape_p; adj = false)
  n, m = shape
  n_p, m_p = shape_p
  k1, k2 = size(kernel)
  r, c = div(n_p - k1, 2), div(m_p - k2, 2)
  K = zeros(n_p, m_p)
  K[(r + 1):(r + k1), (c + 1):(c + k2)] .= kernel
  K̂ = fft(ifftshift(K))
  X = zeros(n_p, m_p)
  X[(n_p - n + 1):end, (m_p - m + 1):end] .= reshape(x, n, m)
  Y = real(ifft(fft(X) .* (adj ? conj.(K̂) : K̂)))
  return vec(Y[(n_p - n + 1):end, (m_p - m + 1):end])
end

@testset "denoising operators" begin
  # non-square image with different paddings in each dimension
  shape, shape_p, k, σ = (32, 48), (38, 52), 3, 1.2
  nm = prod(shape)
  gaussian = [exp(-(i^2 + j^2) / (2σ^2)) for i ∈ (-k):k, j ∈ (-k):k]
  uniform = ones(2k + 1, 2k + 1)
  cases = (
    ("Gaussian", gaussian / sum(gaussian), generate_gaussian_blur(shape, shape_p, k, σ)),
    ("uniform", uniform / sum(uniform), generate_uniform_blur(shape, shape_p, k)),
  )

  for (name, kernel, (H!, H_T!, W!, W_T!)) in cases
    @testset "$name blur" begin
      x, v = rand(nm), rand(nm)
      y, z = similar(x), similar(x)

      @test H!(y, x) ≈ reference_blur(x, kernel, shape, shape_p)
      @test H_T!(y, x) ≈ reference_blur(x, kernel, shape, shape_p, adj = true)
      @test dot(H!(y, x), v) ≈ dot(x, H_T!(z, v))

      # the kernel is nonnegative and sums to one
      @test all(H!(y, x) .≥ -sqrt(eps()))
      Y = reshape(H!(y, ones(nm)), shape)
      @test all(Y[(k + 3):(end - k - 2), (k + 3):(end - k - 2)] .≈ 1)

      # W! is the orthogonal Haar transform and W_T! its inverse
      @test W!(y, x) ≈ vec(dwt(reshape(x, shape), wavelet(WT.haar), 4))
      @test norm(W!(y, x)) ≈ norm(x)
      @test W_T!(z, W!(y, x)) ≈ x
      @test dot(W!(y, x), v) ≈ dot(x, W_T!(z, v))

      @test allocs_op(H!, y, x) == 0
      @test allocs_op(H_T!, y, x) == 0
      # Wavelets.dwt! and idwt! allocate a few small work vectors, but nothing of image size
      @test allocs_op(W!, y, x) < sizeof(x)
      @test allocs_op(W_T!, y, x) < sizeof(x)

      @test_throws DimensionMismatch H!(y, rand(nm + 1))
      @test_throws DimensionMismatch H_T!(rand(nm - 1), x)
    end
  end

  @test_throws ArgumentError generate_gaussian_blur(shape, (30, 52), k)  # padding too small
  @test_throws ArgumentError generate_uniform_blur((36, 48), (40, 52), k)  # 36 not divisible by 2^4
end

@testset "denoising_model" begin
  model, x_true = denoising_model()
  nvar = 256 * 256
  @test model isa NLPModel
  @test model.meta.nvar == nvar
  @test model.meta.name == "denoising"
  @test typeof(x_true) == typeof(model.meta.x0)
  @test length(x_true) == nvar
  @test all(0 .≤ x_true .≤ 1)

  x = model.meta.x0
  f = obj(model, x)
  g = grad(model, x)
  @test f isa Float64
  @test isfinite(f) && f > 0
  @test g isa Vector{Float64}
  @test length(g) == nvar

  fg, gg = objgrad(model, x)
  @test fg ≈ f
  @test gg ≈ g

  # gradient against a central finite difference along a random direction
  d = randn(nvar)
  d ./= norm(d)
  ε = 1e-6
  fd = (obj(model, x + ε * d) - obj(model, x - ε * d)) / (2ε)
  @test isapprox(fd, dot(g, d), atol = 1e-6 * norm(g))

  # without noise, the wavelet transform of the clean image is a global minimizer
  model0, x_true0 = denoising_model(noise_std = 0)
  _, _, W!, _ = generate_gaussian_blur((256, 256), (260, 260), 9, 1.5)
  x_star = W!(similar(x_true0), x_true0)
  @test obj(model0, x_star) ≈ 0 atol = 1e-12
  @test norm(grad(model0, x_star)) ≈ 0 atol = 1e-8
  # with noise, it is still much better than the blurred and noisy starting point
  @test obj(model, x_star) < obj(model, x) / 10

  gx = similar(x)
  @test allocs_obj(model, x) < sizeof(x)
  @test allocs_grad(model, x, gx) < sizeof(x)
  @test allocs_objgrad(model, x, gx) < sizeof(x)

  model32, x_true32 = denoising_model(T = Float32)
  x32 = model32.meta.x0
  @test eltype(x32) == Float32
  @test eltype(x_true32) == Float32
  @test obj(model32, x32) isa Float32
  @test grad(model32, x32) isa Vector{Float32}

  # the problem is defined for the size of the cameraman image only
  @test_throws ArgumentError denoising_model((128, 128), (132, 132))
end
