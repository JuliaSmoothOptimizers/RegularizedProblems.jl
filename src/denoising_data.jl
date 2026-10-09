export generate_uniform_blur, generate_gaussian_blur

#=
Blur and wavelet operators for the image denoising problem described in

Stella, L., Themelis, A. & Patrinos, P.
Forward–backward quasi-Newton methods for nonsmooth optimization problems.
Comput Optim Appl 67, 443–487 (2017). https://doi.org/10.1007/s10589-017-9912-y

and adapted from the original implementation in Python by the authors of the paper:

Chouzenoux, E., Martin, S. & Pesquet, JC.
A Local MM Subspace Method for Solving Constrained Variational Problems in Image Recovery.
J Math Imaging Vis 65, 253–276 (2023). https://doi.org/10.1007/s10851-022-01112-z
=#

# Normalized (2k+1) × (2k+1) Gaussian kernel with standard deviation `kernel_sigma`.
function gaussian_kernel(::Type{T}, kernel_size::Integer, kernel_sigma::Real) where {T}
  r = (-kernel_size):kernel_size
  s = 2 * T(kernel_sigma)^2
  kernel = [exp(-(i^2 + j^2) / s) for i ∈ r, j ∈ r]
  kernel ./= sum(kernel)
  return kernel
end

# Normalized (2k+1) × (2k+1) uniform (box) kernel.
uniform_kernel(::Type{T}, kernel_size::Integer) where {T} =
  fill(one(T) / (2 * kernel_size + 1)^2, 2 * kernel_size + 1, 2 * kernel_size + 1)

# Real-FFT coefficients of the blur kernel, zero-padded to `shape_p` and circularly shifted
# so that its center sits at the origin.
# The 1 / (n_p * m_p) normalization of the inverse FFT is folded into the coefficients so that
# the unnormalized backward transform `brfft` can be used when applying the operator.
function blur_spectrum(kernel::AbstractMatrix{T}, shape_p) where {T}
  n_p, m_p = shape_p
  k1, k2 = size(kernel)
  r = div(n_p - k1, 2)
  c = div(m_p - k2, 2)
  K = zeros(T, n_p, m_p)
  K[(r + 1):(r + k1), (c + 1):(c + k2)] .= kernel
  spectrum = FFTW.rfft(FFTW.ifftshift(K))
  spectrum ./= n_p * m_p
  return spectrum
end

function blur_and_wavelet_operators(
  kernel::AbstractMatrix{T},
  shape,
  shape_p,
  levels::Integer,
) where {T <: Union{Float32, Float64}}
  n, m = shape
  n_p, m_p = shape_p
  (n_p ≥ n && m_p ≥ m) || throw(
    ArgumentError("the padded shape $shape_p must be at least as large as the image shape $shape"),
  )
  (n_p ≥ size(kernel, 1) && m_p ≥ size(kernel, 2)) || throw(
    ArgumentError("the padded shape $shape_p is too small for a kernel of size $(size(kernel))"),
  )
  (n % 2^levels == 0 && m % 2^levels == 0) || throw(
    ArgumentError(
      "the image dimensions $shape must be divisible by 2^$levels for a $levels-level wavelet transform",
    ),
  )

  nm = n * m
  a1 = n_p - n  # number of zero rows padded above the image
  a2 = m_p - m  # number of zero columns padded to the left of the image

  spectrum = blur_spectrum(kernel, shape_p)
  spectrum_adj = conj.(spectrum)  # spectrum of the adjoint (flipped) kernel

  # Work arrays and FFT plans shared by H! and H_T!.
  xpad = zeros(T, n_p, m_p)
  xhat = similar(spectrum)
  fwd = FFTW.plan_rfft(xpad)
  bwd = FFTW.plan_brfft(xhat, n_p)

  # y = crop(ifft(fft(pad(x)) .* s))
  function blur!(y, x, s)
    length(x) == nm || throw(DimensionMismatch("input has length $(length(x)), expected $nm"))
    length(y) == nm || throw(DimensionMismatch("output has length $(length(y)), expected $nm"))
    fill!(xpad, zero(T))
    @inbounds for j = 1:m, i = 1:n
      xpad[a1 + i, a2 + j] = x[i + (j - 1) * n]
    end
    mul!(xhat, fwd, xpad)
    xhat .*= s
    mul!(xpad, bwd, xhat)  # overwrites xhat as well
    @inbounds for j = 1:m, i = 1:n
      y[i + (j - 1) * n] = xpad[a1 + i, a2 + j]
    end
    return y
  end

  H!(y, x) = blur!(y, x, spectrum)
  H_T!(y, x) = blur!(y, x, spectrum_adj)

  wt = Wavelets.wavelet(Wavelets.WT.haar)

  function W!(y, x)
    Wavelets.dwt!(reshape_array(y, (n, m)), reshape_array(x, (n, m)), wt, levels)
    return y
  end

  function W_T!(y, x)
    Wavelets.idwt!(reshape_array(y, (n, m)), reshape_array(x, (n, m)), wt, levels)
    return y
  end

  return H!, H_T!, W!, W_T!
end

"""
    H!, H_T!, W!, W_T! = generate_gaussian_blur(shape, shape_p, kernel_size, kernel_sigma = 1.5; T = Float64, levels = 4)

Return in-place linear operators used to model image deblurring with a Gaussian blur.

Each operator is called as `op!(y, x)` and overwrites `y` with the image of `x`, where `x` and
`y` are vectors of length `prod(shape)` that store a `shape` image column by column:

* `H!(y, x)` blurs `x` with a `(2 kernel_size + 1) × (2 kernel_size + 1)` Gaussian kernel of
  standard deviation `kernel_sigma`: the image is zero-padded to `shape_p`, circularly convolved
  with the kernel and cropped back to `shape`;
* `H_T!(y, x)` applies the adjoint of `H!`;
* `W!(y, x)` applies an orthogonal `levels`-level 2D Haar wavelet transform;
* `W_T!(y, x)` applies the inverse, which is also the adjoint, of `W!`.

The operators reuse preallocated work arrays and FFT plans, so they do not allocate arrays of the
size of the image, but they are not thread safe.
For `W!` and `W_T!`, `y` and `x` must be different arrays.

## Keyword arguments

* `T`: floating-point type, `Float64` or `Float32`;
* `levels`: number of levels of the wavelet transform; each dimension of `shape` must be divisible by `2^levels`.
"""
function generate_gaussian_blur(
  shape,
  shape_p,
  kernel_size::Integer,
  kernel_sigma::Real = 1.5;
  T::Type{<:Union{Float32, Float64}} = Float64,
  levels::Integer = 4,
)
  kernel = gaussian_kernel(T, kernel_size, kernel_sigma)
  return blur_and_wavelet_operators(kernel, shape, shape_p, levels)
end

"""
    H!, H_T!, W!, W_T! = generate_uniform_blur(shape, shape_p, kernel_size; T = Float64, levels = 4)

Same as [`generate_gaussian_blur`](@ref), but `H!` blurs with a uniform (box) kernel of size
`(2 kernel_size + 1) × (2 kernel_size + 1)`.
"""
function generate_uniform_blur(
  shape,
  shape_p,
  kernel_size::Integer;
  T::Type{<:Union{Float32, Float64}} = Float64,
  levels::Integer = 4,
)
  kernel = uniform_kernel(T, kernel_size)
  return blur_and_wavelet_operators(kernel, shape, shape_p, levels)
end
