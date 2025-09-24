using RegularizedProblems
using Documenter

DocMeta.setdocmeta!(
  RegularizedProblems,
  :DocTestSetup,
  :(using RegularizedProblems);
  recursive = true,
)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
  file for file in readdir(joinpath(@__DIR__, "src")) if
  file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
  modules = [RegularizedProblems],
  authors = "Dominique Orban <dominique.orban@gmail.com>, Robert Baraldi <robertjbaraldi@gmail.com",
  repo = "https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl/blob/{commit}{path}#{line}",
  sitename = "RegularizedProblems.jl",
  format = Documenter.HTML(;
    canonical = "https://JuliaSmoothOptimizers.github.io/RegularizedProblems.jl",
  ),
  pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/JuliaSmoothOptimizers/RegularizedProblems.jl")
