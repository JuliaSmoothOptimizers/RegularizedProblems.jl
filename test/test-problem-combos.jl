# Test that we can at least instantiate the models
rnlp, rnls = setup_bpdn_l0()
@test isa(rnlp, RegularizedNLPModel)
@test isa(rnls, RegularizedNLSModel)
