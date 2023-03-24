using Documenter
using ADNLPModels, DifferentialEquations, MLDatasets
using RegularizedProblems

makedocs(
  modules = [RegularizedProblems],
  doctest = true,
  # linkcheck = true,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "RegularizedProblems.jl",
  pages = Any["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/RegularizedProblems.jl.git",
  push_preview = true,
  devbranch = "main",
)
