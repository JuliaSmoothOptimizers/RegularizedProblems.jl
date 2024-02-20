using Images
using FFTW
using Wavelets
export denoising_model

include("denoising_data.jl")

function denoising_model(shape, shape_p, KERNEL_SIZE, KERNEL_SIGMA = 1.5)
  sigma = 10^-3
  data_path = joinpath(@__DIR__, "..", "images/cameraman.png")
  cameraman_image = Images.load(data_path)
  x_t = vec(Float64.(cameraman_image))
  H, H_T, W, W_T = generate_gaussian_blur(shape, shape_p, KERNEL_SIZE, KERNEL_SIGMA)
  (n, m) = shape
  b = H(x_t) + randn(n * m) * sigma
  y = similar(b)
  z = similar(b)

  function obj(x)
    y .= W_T(x)
    y .= H(y)
    z .= log.((y .- b) .^ 2 .+ 1)
    return sum(z)
  end

  function grad!(g, x)
    y .= H(W_T(x))
    z .= 1 ./ ((y .- b) .^ 2 .+ 1)
    @. z = 2 * z * (y - b)
    g .= W(H_T(z))
    return g
  end

  x0 = W(b)

  FirstOrderModel(obj, grad!, x0, name = "denoing_model"), x_t
end
