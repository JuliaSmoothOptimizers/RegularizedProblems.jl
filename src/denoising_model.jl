export denoising_model

include("denoising_data.jl")

"""
    model, x_true = denoising_model(shape = (256, 256), shape_p = (260, 260), kernel_size = 9, kernel_sigma = 1.5; T = Float64, noise_std = 1e-3)

Image deblurring and denoising problem of Stella, Themelis and Patrinos (2017)

    minimize f(x) = ∑ᵢ log(((H Wᵀ x - b)ᵢ)² + 1),

where `x` contains the 2D Haar wavelet coefficients of an image, `Wᵀ` is the inverse wavelet
transform, `H` is a Gaussian blur and `b = H x_true + ε` is a blurred and noisy observation of
the 256 × 256 cameraman image `x_true`, with `ε ~ N(0, noise_std² I)`.
The loss is smooth but nonconvex. The problem is meant to be combined with a regularizer that
promotes sparsity of the wavelet coefficients, such as `λ‖x‖₁` or `λ‖x‖₀`.

## Arguments

* `shape`: size of the image, which must be the size of the cameraman image, `(256, 256)`;
* `shape_p`: size of the zero-padded image on which the circular convolution is computed;
* `kernel_size`: the blur kernel has size `(2 kernel_size + 1) × (2 kernel_size + 1)`;
* `kernel_sigma`: standard deviation of the Gaussian blur kernel.

## Keyword arguments

* `T`: floating-point type, `Float64` or `Float32`;
* `noise_std`: standard deviation of the additive Gaussian noise.

## Return values

* `model`: an `NLPModel` whose objective, gradient and `objgrad!` work in place and do not
  allocate arrays of the size of the image, and whose starting point is the wavelet transform of `b`;
* `x_true`: the clean image, stored column by column in pixel space.
  The corresponding point in the variable space of `model` is its wavelet transform `W x_true`,
  see [`generate_gaussian_blur`](@ref).

This problem is only available once FFTW, Images and Wavelets have been loaded.
"""
function denoising_model(
  shape = (256, 256),
  shape_p = (260, 260),
  kernel_size::Integer = 9,
  kernel_sigma::Real = 1.5;
  T::Type{<:Union{Float32, Float64}} = Float64,
  noise_std::Real = 1e-3,
)
  image = Images.load(joinpath(@__DIR__, "..", "images", "cameraman.png"))
  size(image) == Tuple(shape) || throw(
    ArgumentError("shape must be $(size(image)), the size of the cameraman image, got $shape"),
  )
  x_true = vec(T.(image))
  H!, H_T!, W!, W_T! = generate_gaussian_blur(shape, shape_p, kernel_size, kernel_sigma; T = T)

  b = H!(similar(x_true), x_true)
  b .+= T(noise_std) .* randn(T, length(b))
  u = similar(b)  # work array in pixel space
  r = similar(b)  # residual H Wᵀ x - b

  function residual!(x)
    W_T!(u, x)
    H!(r, u)
    r .-= b
    return r
  end

  loss(r) = sum(ri -> log1p(ri^2), r)

  # Overwrite r with the derivative of the loss and return the gradient W Hᵀ r in g.
  function backprop!(g)
    @. r = 2 * r / (r^2 + 1)
    H_T!(u, r)
    W!(g, u)
    return g
  end

  function obj(x)
    residual!(x)
    return loss(r)
  end

  function grad!(g, x)
    residual!(x)
    return backprop!(g)
  end

  function objgrad!(g, x)
    residual!(x)
    f = loss(r)
    return f, backprop!(g)
  end

  x0 = W!(similar(b), b)
  model = NLPModel(x0, obj; grad = grad!, objgrad = objgrad!, meta_args = (name = "denoising",))
  return model, x_true
end
