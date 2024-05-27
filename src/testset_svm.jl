# Predefine a set of common problem instances.
export setup_svm_train_lhalf,
  setup_svm_test_lhalf, setup_svm_train_l0, setup_svm_test_l0, setup_svm_train_l1, setup_svm_test_l1

function setup_svm_train_lhalf(args...; kwargs...)
  model, nls_model, _ = svm_train_model(args...)
  h = ShiftedProximalOperators.RootNormLhalf(0.1)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_svm_test_lhalf(args...; kwargs...)
  model, nls_model, _ = svm_test_model(args...)
  h = ShiftedProximalOperators.RootNormLhalf(0.1)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_svm_train_l0(args...; kwargs...)
  model, nls_model, _ = svm_train_model(args...)
  h = ProximalOperators.NormL0(0.1)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_svm_test_l0(args...; kwargs...)
  model, nls_model, _ = svm_test_model(args...)
  h = ProximalOperators.NormL0(0.1)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_svm_train_l1(args...; kwargs...)
  model, nls_model, _ = svm_train_model(args...)
  h = ProximalOperators.NormL1(0.1)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end

function setup_svm_test_l1(args...; kwargs...)
  model, nls_model, _ = svm_test_model(args...)
  h = ProximalOperators.NormL1(0.1)
  return RegularizedNLPModel(model, h), RegularizedNLSModel(nls_model, h)
end
