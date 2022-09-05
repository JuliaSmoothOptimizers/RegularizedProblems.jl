var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [RegularizedProblems]","category":"page"},{"location":"reference/#RegularizedProblems.FirstOrderModel","page":"Reference","title":"RegularizedProblems.FirstOrderModel","text":"model = FirstOrderModel(f, ∇f!; name = \"first-order model\")\n\nA simple subtype of AbstractNLPModel to represent a smooth objective.\n\nArguments\n\nf :: F <: Function: a function such that f(x) returns the objective value at x;\n∇f! :: G <: Function: a function such that ∇f!(g, x) stores the gradient of the objective at x in g;\nx :: AbstractVector: an initial guess.\n\nKeyword arguments\n\nAll keyword arguments are passed through to the NLPModelMeta constructor.\n\n\n\n\n\n","category":"type"},{"location":"reference/#RegularizedProblems.FirstOrderNLSModel","page":"Reference","title":"RegularizedProblems.FirstOrderNLSModel","text":"model = FirstOrderNLSModel(r!, jv!, jtv!; name = \"first-order NLS model\")\n\nA simple subtype of AbstractNLSModel to represent a nonlinear least-squares problem with a smooth residual.\n\nArguments\n\nr! :: R <: Function: a function such that r!(y, x) stores the residual at x in y;\njv! :: J <: Function: a function such that jv!(u, x, v) stores the product between the residual Jacobian at x and the vector v in u;\njtv! :: Jt <: Function: a function such that jtv!(u, x, v) stores the product between the transpose of the residual Jacobian at x and the vector v in u;\nx :: AbstractVector: an initial guess.\n\nKeyword arguments\n\nAll keyword arguments are passed through to the NLPModelMeta constructor.\n\n\n\n\n\n","category":"type"},{"location":"reference/#RegularizedProblems.bpdn_model-Tuple","page":"Reference","title":"RegularizedProblems.bpdn_model","text":"model, nls_model, sol = bpdn_model(args...)\nmodel, nls_model, sol = bpdn_model(compound = 1, args...)\n\nReturn an instance of an NLPModel and an instance of an NLSModel representing the same basis-pursuit denoise problem, i.e., the under-determined linear least-squares objective\n\n½ ‖Ax - b‖₂²,\n\nwhere A has orthonormal rows and b = A * x̄ + ϵ, x̄ is sparse and ϵ is a noise vector following a normal distribution with mean zero and standard deviation σ.\n\nArguments\n\nm :: Int: the number of rows of A\nn :: Int: the number of columns of A (with n ≥ m)\nk :: Int: the number of nonzero elements in x̄\nnoise :: Float64: noise standard deviation σ (default: 0.01).\n\nThe second form calls the first form with arguments\n\nm = 200 * compound\nn = 512 * compound\nk =  10 * compound\n\nKeyword arguments\n\nbounds :: Bool: whether or not to include nonnegativity bounds in the model (default: false).\n\nReturn Value\n\nAn instance of a FirstOrderModel and of a FirstOrderNLSModel that represent the same basis-pursuit denoise problem, and the exact solution x̄.\n\nIf bounds == true, the positive part of x̄ is returned.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RegularizedProblems.fh_model-Tuple{}","page":"Reference","title":"RegularizedProblems.fh_model","text":"fh_model(; kwargs...)\n\nReturn an instance of an NLPModel and an instance of an NLSModel representing the same Fitzhugh-Nagumo problem, i.e., the over-determined nonlinear least-squares objective\n\n½ ‖F(x)‖₂²,\n\nwhere F: ℝ⁵ → ℝ²⁰² represents the fitting error between a simulation of the Fitzhugh-Nagumo model with parameters x and a simulation of the Van der Pol oscillator with fixed, but unknown, parameters.\n\nKeyword Arguments\n\nAll keyword arguments are passed directly to the ADNLPModel (or ADNLSModel) constructure, e.g., to set the automatic differentiation backend.\n\nReturn Value\n\nAn instance of an ADNLPModel that represents the Fitzhugh-Nagumo problem, an instance of an ADNLSModel that represents the same problem, and the exact solution.\n\n\n\n\n\n","category":"method"},{"location":"#RegularizedProblems","page":"Home","title":"RegularizedProblems","text":"","category":"section"},{"location":"#Synopsis","page":"Home","title":"Synopsis","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides sameple problems suitable for developing and testing first and second-order methods for regularized optimization, i.e., they have the general form","category":"page"},{"location":"","page":"Home","title":"Home","text":"min_x in mathbbR^n  f(x) + h(x)","category":"page"},{"location":"","page":"Home","title":"Home","text":"where f mathbbR^n to mathbbR has Lipschitz-continuous gradient and h mathbbR^n to mathbbR cup infty is lower semi-continuous and proper. The smooth term f describes the objective to minimize while the role of the regularizer h is to select a solution with desirable properties: minimum norm, sparsity below a certain level, maximum sparsity, etc.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Models for f are instances of NLPModels and often represent nonlinear least-squares residuals, i.e., f(x) = tfrac12 F(x)_2^2 where F mathbbR^n to mathbbR^m.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The regularizer h should be obtained from ProximalOperators.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The final regularized problem is intended to be solved by way of solver for nonsmooth regularized optimization such as those in RegularizedOptimization.jl.","category":"page"},{"location":"#Problems-implemented","page":"Home","title":"Problems implemented","text":"","category":"section"},{"location":"#Basis-pursuit-denoise","page":"Home","title":"Basis-pursuit denoise","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Calling model = bpdn_model() returns a model representing the smooth underdetermined linear least-squares residual","category":"page"},{"location":"","page":"Home","title":"Home","text":"f(x) = tfrac12 Ax - b_2^2","category":"page"},{"location":"","page":"Home","title":"Home","text":"where A has orthonormal rows. The right-hand side is generated as b = A x_star + varepsilon where x_star is a sparse vector, varepsilon sim mathcalN(0 sigma) and sigma in (0 1) is a fixed noise level.","category":"page"},{"location":"","page":"Home","title":"Home","text":"When solving the basis-pursuit denoise problem, the goal is to recover x approx x_star. In particular, x should have the same sparsity pattern as x_star. That is typically accomplished by choosing a regularizer of the form","category":"page"},{"location":"","page":"Home","title":"Home","text":"h(x) = lambda x_1 for a well-chosen lambda  0;\nh(x) = x_0;\nh(x) = chi(x k mathbbB_0) for k approx x_star_0;","category":"page"},{"location":"","page":"Home","title":"Home","text":"where chi(x k mathbbB_0) is the indicator of the ell_0-pseudonorm ball of radius k.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Calling model = bpdn_nls_model() returns the same problem modeled explicitly as a least-squares problem.","category":"page"},{"location":"#Fitzhugh-Nagumo-data-fitting-problem","page":"Home","title":"Fitzhugh-Nagumo data-fitting problem","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If ADNLPModels and DifferentialEquations have been imported, model = fh_model() returns a model representing the over-determined nonlinear least-squares residual","category":"page"},{"location":"","page":"Home","title":"Home","text":"f(x) = tfrac12 F(x)_2^2","category":"page"},{"location":"","page":"Home","title":"Home","text":"where F mathbbR^5 to mathbbR^202 represents the residual between a simulation of the Fitzhugh-Nagumo system with parameters x and a simulation of the Van der Pol oscillator with preset, but unknown, parameters x_star.","category":"page"},{"location":"","page":"Home","title":"Home","text":"A feature of the Fitzhugh-Nagumo model is that it reduces to the Van der Pol oscillator when certain parameters are set to zero. Thus here again, the objective is to recover a sparse solution to the data-fitting problem. Hence, typical regularizers are the same as those used for the basis-pursuit denoise problem.","category":"page"}]
}
