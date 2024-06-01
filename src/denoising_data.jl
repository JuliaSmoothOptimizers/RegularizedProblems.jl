export generate_HW

# Function to unpad an array
function unpad(x, n_p, m_p, n)
  a = n_p - n
  x = reshape_array(x, (n_p, m_p))
  unpadded_x = x[(a + 1):end, (a + 1):end]
  return unpadded_x
end

# Function to pad an array
function pad(z, s1, s2)
  x = cat(zeros(size(z, 1), s1), z, dims = 2)
  x = cat(x, zeros(size(z, 1), s2), dims = 2)
  x = cat(x, zeros(s2, size(x, 2)), dims = 1)
  x = cat(zeros(s1, size(x, 2)), x, dims = 1)
  return x
end

# Function to pad a specific array to match suitable dimensions
function pad_x(z, a)
  t = cat(zeros(size(z, 1), a), z, dims = 2)
  t = cat(zeros(a, size(z, 2) + a), t, dims = 1)
  return t
end

# Function to create a meshgrid for the kernel
function meshgrid(x_range, y_range)
  m = length(x_range)
  n = length(y_range)
  x = repeat(x_range, inner = (1, n))
  y = repeat(y_range', outer = (m, 1))
  return x, y
end

# Function to generate a Gaussian kernel
function my_gaussian_kernel(kernel_size, kernel_sigma)
  x, y = meshgrid((-kernel_size):kernel_size, (-kernel_size):kernel_size)
  normal = 1 / (2.0 * pi * kernel_sigma^2)
  kernel = exp.(-((x .^ 2 + y .^ 2) / (2.0 * kernel_sigma^2))) * normal
  kernel = kernel / sum(kernel)
  return kernel
end

# Function to generate a uniform kernel
function uniform_kernel(kernel_size)
  kernel = ones(2 * kernel_size + 1, 2 * kernel_size + 1)
  kernel = kernel / sum(kernel)
  return kernel
end

# Main function to generate H, H_T, W, and W_T functions
function generate_HW(shape, shape_p, KERNEL_SIZE, KERNEL_TYPE, KERNEL_SIGMA = 1.5)
  (n, m) = shape
  (n_p, m_p) = shape_p
  a = n_p - n

  if KERNEL_TYPE == "gaussian"
    kernel_h = my_gaussian_kernel(KERNEL_SIZE, KERNEL_SIGMA)
  elseif KERNEL_TYPE == "uniform"
    kernel_h = uniform_kernel(KERNEL_SIZE)
  else
    println("This KERNEL_TYPE is not defined")
  end

  sz = (n_p - (2 * KERNEL_SIZE + 1), m_p - (2 * KERNEL_SIZE + 1))
  kernel_h = pad(kernel_h, div(sz[1], 2), div(sz[1], 2) + 1)
  kernel_h = ifftshift(kernel_h)
  fft_h = fft(kernel_h)

  # Function H: Applies a linear transformation H which models the blur to an input x
  function H(x)
    x = reshape_array(x, (n, m))
    x = pad_x(x, a)
    fft_x = fft(x)
    x_new = real(ifft(fft_x .* fft_h))
    return unpad(x_new, n_p, m_p, n)[:]
  end

  # Function H_T: Applies the transpose of the linear transformation H to an input x
  function H_T(x)
    x = reshape_array(x, (n, m))
    x = pad_x(x, a)
    fft_x = fft(x)
    x_new = real(ifft(fft_x .* conj.(fft_h)))
    return unpad(x_new, n_p, m_p, n)[:]
  end

  wt = wavelet(WT.haar)

  # ----- Discrete Wavelet Transform (DWT) -----

  # Function W: Applies the DWT to an input x
  function W(x)
    return dwt(reshape_array(x, (n, m)), wt, 4)[:]
  end

  # Function W_T: Applies the inverse DWT to an input x
  function W_T(x)
    return idwt(reshape_array(x, (n, m)), wt, 4)[:]
  end

  # Return the generated functions
  return H, H_T, W, W_T
end
