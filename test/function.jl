@test onehot(5, 1) == [1.0, 0.0, 0.0, 0.0, 0.0]
@test sigmoid.f(0) == 0.5