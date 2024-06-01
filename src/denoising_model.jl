export denoising_model

include("denoising_data.jl")

function denoing_model(shape, shape_p, KERNEL_SIZE, KERNEL_TYPE, KERNEL_SIGMA = 1.5)
  sigma = 10^-3
  cameraman_image = load("images/cameraman.png")
  x_t = vec(Float64.(cameraman_image))
  H, H_T, W, W_T = generate_HW(shape, shape_p, KERNEL_SIZE, KERNEL_TYPE, KERNEL_SIGMA)
  (n, m) = shape
  b = H(x_t) + randn(n * m) * sigma
  y = similar(b)
  z = similar(b)

  function obj(x)
    y .= W_T(x)
    y .= H(y)
    z .= log.((y - b) .^ 2 .+ 1)
    return sum(z)
  end

  function grad!(g, x)
    y = H(W_T(x))
    term = 1.0 ./ ((y - b) .^ 2 .+ 1)
    z = term .* (y - b)
    z .*= 2
    g .= W(H_T(z))
    return g
  end

  x0 = W(b)

  FirstOrderModel(obj, grad!, x0, name = "denoing_model"), x_t
end
