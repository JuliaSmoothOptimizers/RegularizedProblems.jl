export abcd_problems, abcd_problem_info

# Test problems of
#
#   R. Lopes, S. A. Santos and P. J. S. Silva,
#   Accelerating block coordinate descent methods with identification strategies,
#   Computational Optimization and Applications 72, 609–640 (2019),
#   https://doi.org/10.1007/s10589-018-00056-8
#
# The data (archive `Data-Files.zip`, ≈ 3.2 GB) is distributed by the authors at
# https://drive.google.com/drive/folders/1yUqNvwDazKED-Umfe5O95WP-JyedkLnm
#
# Each entry is (label, dataset, source, nrow, ncol) as reported in Tables 1–3 of the paper.
# The labels SRi / SCi refer to Tables 2 / 3 and NNi to Table 1.
const _ABCD_TABLE = [
  # Table 2: more rows than columns
  ("SR1", "a1a.t", "LIBSVM", 30956, 123),
  ("SR2", "a2a.t", "LIBSVM", 30296, 123),
  ("SR3", "a4a.t", "LIBSVM", 27780, 123),
  ("SR4", "connect-4", "LIBSVM", 67557, 126),
  ("SR5", "dna.scale", "LIBSVM", 2000, 180),
  ("SR6", "mnist", "LIBSVM", 60000, 717),
  ("SR7", "mushrooms", "LIBSVM", 8124, 112),
  ("SR8", "phishing", "LIBSVM", 11055, 68),
  ("SR9", "protein", "LIBSVM", 17766, 356),
  ("SR10", "w2a", "LIBSVM", 3470, 293),
  ("SR11", "w4a.t", "LIBSVM", 42383, 300),
  ("SR12", "w5a.t", "LIBSVM", 39861, 300),
  ("SR13", "w6a.t", "LIBSVM", 32561, 300),
  ("SR14", "w8a.t", "LIBSVM", 14951, 300),
  ("SR15", "w5a", "LIBSVM", 2833, 299),
  ("SR16", "real-sim", "LIBSVM", 72309, 20958),
  ("SR17", "rcv1-test-bin", "LIBSVM", 677399, 42735),
  ("SR18", "rcv1-test-mult", "LIBSVM", 518571, 41400),
  ("SR19", "J-Lee", "Komarek", 181395, 105353),
  ("SR20", "webspam-unigram", "LIBSVM", 350000, 138),
  ("SR21", "Maragal-4", "SuiteSparse", 1964, 1027),
  ("SR22", "Maragal-5", "SuiteSparse", 4654, 3296),
  ("SR23", "Maragal-6", "SuiteSparse", 21255, 10144),
  ("SR24", "Maragal-7", "SuiteSparse", 46845, 26525),
  # Table 3: more columns than rows
  ("SC1", "peppers05-6-6", "Bradley et al. (2011)", 32768, 65536),
  ("SC2", "peppers05-12-12", "Bradley et al. (2011)", 32768, 65536),
  ("SC3", "peppers025-12-12", "Bradley et al. (2011)", 16384, 65536),
  ("SC4", "SparcoProblem401", "SPARCO", 29166, 57344),
  ("SC5", "SparcoProblem402", "SPARCO", 29166, 57344),
  ("SC6", "SparcoProblem603", "SPARCO", 1024, 4096),
  ("SC7", "finance1000", "Bradley et al. (2011)", 30465, 216842),
  ("SC8", "dbworld-bodies", "UCI", 64, 4702),
  ("SC9", "dexter-train", "UCI", 300, 7751),
  ("SC10", "dexter-valid", "UCI", 300, 7847),
  ("SC11", "dorothea-train", "UCI", 800, 88119),
  ("SC12", "dorothea-valid", "UCI", 350, 72113),
  ("SC13", "news20-binary", "LIBSVM", 19996, 1355191),
  ("SC14", "news20-scale", "LIBSVM", 15935, 60346),
  ("SC15", "news20-t-scale", "LIBSVM", 3993, 39128),
  ("SC16", "rcv1-train-bin", "LIBSVM", 20242, 44504),
  ("SC17", "rcv1-train-mult", "LIBSVM", 15564, 36842),
  ("SC18", "sector-scale", "LIBSVM", 6412, 49087),
  ("SC19", "sector-t-scale", "LIBSVM", 3207, 39234),
  ("SC20", "Blanc-Mel", "Komarek", 186414, 685568),
  ("SC21", "farm-ads-vect", "UCI", 4143, 54877),
  ("SC22", "mug05-12-12", "Bradley et al. (2011)", 12410, 24820),
  ("SC23", "mug025-12-12", "Bradley et al. (2011)", 6205, 24820),
  ("SC24", "mug075-12-12", "Bradley et al. (2011)", 13651, 24820),
  ("SC25", "Maragal-8", "SuiteSparse", 33212, 60845),
  # Table 1: nonnegative least squares instances
  ("NN1", "illc1033", "Matrix Market", 1033, 320),
  ("NN2", "illc1850", "Matrix Market", 1850, 712),
  ("NN3", "well1033", "Matrix Market", 1033, 320),
  ("NN4", "well1850", "Matrix Market", 1850, 712),
  ("NN5", "real-sim", "LIBSVM", 72309, 20958),
  ("NN6", "mnist", "LIBSVM", 60000, 717),
  ("NN7", "webspam-unigram", "LIBSVM", 350000, 138),
  ("NN8", "Maragal-3", "SuiteSparse", 1690, 858),
  ("NN9", "Maragal-4", "SuiteSparse", 1964, 1027),
  ("NN10", "Maragal-5", "SuiteSparse", 4654, 3296),
  ("NN11", "Maragal-6", "SuiteSparse", 21255, 10144),
  ("NN12", "Maragal-7", "SuiteSparse", 46845, 26525),
]

const _ABCD_INFO = Dict(
  t[1] => (dataset = t[2], source = t[3], nrow = t[4], ncol = t[5]) for t in _ABCD_TABLE
)

_abcd_names(prefix, indices) = ["$prefix$i" for i in indices]

"""
    names = abcd_problems(set = :lasso)

Return the names of the problems of the data set of Lopes, Santos and Silva (2019)
that belong to `set`. Names can be passed to `abcd_lasso_model`, `abcd_logistic_model`
or `abcd_data` once `MAT` has been loaded.

The possible values of `set` are

* `:lasso`: the 49 Lasso problems `SR1`–`SR24` and `SC1`–`SC25` (Tables 2 and 3 of the paper);
* `:lasso_tuning`: the 18 Lasso problems used to tune parameters (`SR1`–`SR9`, `SC1`–`SC9`);
* `:lasso_test`: the 31 Lasso problems used in the comparisons (`SR10`–`SR24`, `SC10`–`SC25`);
* `:logistic`: the 35 ℓ₁-regularized logistic regression problems used in the paper;
* `:logistic_all`: all 49 files of `Data-Logistic`, including the 14 whose randomly generated
  labels were rejected by the authors (marked ***** in the paper);
* `:logistic_tuning`: the 12 logistic problems used to tune parameters;
* `:logistic_test`: the 23 logistic problems used in the comparisons;
* `:nonneg_lasso`: the 12 nonnegative Lasso problems `NN1`–`NN12` (Table 1).

Logistic problems are named `SRlogi` and `SClogi`, as the files in `Data-Logistic`.
"""
function abcd_problems(set::Symbol = :lasso)
  if set == :lasso
    return vcat(_abcd_names("SR", 1:24), _abcd_names("SC", 1:25))
  elseif set == :lasso_tuning
    return vcat(_abcd_names("SR", 1:9), _abcd_names("SC", 1:9))
  elseif set == :lasso_test
    return vcat(_abcd_names("SR", 10:24), _abcd_names("SC", 10:25))
  elseif set == :logistic
    return vcat(
      _abcd_names("SRlog", [1:3; 7; 8; 10:20; 22:24]),
      _abcd_names("SClog", [7:21; 25]),
    )
  elseif set == :logistic_all
    return vcat(_abcd_names("SRlog", 1:24), _abcd_names("SClog", 1:25))
  elseif set == :logistic_tuning
    return vcat(_abcd_names("SRlog", [1:3; 8; 10; 11]), _abcd_names("SClog", 8:13))
  elseif set == :logistic_test
    return vcat(_abcd_names("SRlog", [7; 12:20; 22:24]), _abcd_names("SClog", [7; 14:21; 25]))
  elseif set == :nonneg_lasso
    return _abcd_names("NN", 1:12)
  else
    throw(
      ArgumentError(
        "unknown set :$set; use one of :lasso, :lasso_tuning, :lasso_test, :logistic, " *
        ":logistic_all, :logistic_tuning, :logistic_test, :nonneg_lasso",
      ),
    )
  end
end

"""
    info = abcd_problem_info(name)

Return a named tuple describing the problem `name` of the data set of Lopes, Santos and Silva (2019),
with fields

* `name`: the canonical name, which is also the stem of the data file (e.g., `"SRlog10"`);
* `kind`: `:lasso`, `:logistic` or `:nonneg_lasso`;
* `folder`, `file`: location of the data file relative to the data directory;
* `dataset`, `source`: the original data set and where it comes from;
* `nrow`, `ncol`: the dimensions of `A` reported in the paper.

`name` may be a `String` or a `Symbol`. A logistic problem may be referred to as `"SRlog10"`
or, equivalently, `"SR10"` together with `kind = :logistic`.
"""
function abcd_problem_info(
  name::Union{AbstractString, Symbol};
  kind::Union{Symbol, Nothing} = nothing,
)
  s = String(name)
  m = match(r"^(SR|SC)(log)?([0-9]+)$", s)
  if m !== nothing
    base = m.captures[1] * m.captures[3]
    k = m.captures[2] === nothing ? :lasso : :logistic
    if kind !== nothing
      kind ∈ (:lasso, :logistic) ||
        throw(ArgumentError("problem \"$s\" cannot be of kind :$kind"))
      (k == :logistic && kind == :lasso) &&
        throw(ArgumentError("\"$s\" is a logistic problem; use \"$base\" for the Lasso problem"))
      k = kind
    end
  else
    base = s
    k = :nonneg_lasso
  end
  haskey(_ABCD_INFO, base) ||
    throw(ArgumentError("unknown problem \"$s\"; see `abcd_problems()` for valid names"))
  (k == :nonneg_lasso && kind !== nothing && kind != :nonneg_lasso) &&
    throw(ArgumentError("problem \"$s\" cannot be of kind :$kind"))
  stem = k == :logistic ? base[1:2] * "log" * base[3:end] : base
  folder =
    k == :lasso ? "Data-Lasso" : (k == :logistic ? "Data-Logistic" : "Data-Non-Negative-Lasso")
  return merge((name = stem, kind = k, folder = folder, file = stem * ".mat"), _ABCD_INFO[base])
end
