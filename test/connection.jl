# test chain connection and dense layer
model1 = Chain(DenseLayer(20 => 10), DenseLayer(10, 10))
@test size(model1) == ((20,) => (10,))
@test size(model1(rand(20))) == (10,)

# test convolution layer
model2 = Chain(
    ConvLayer((10, 10), (3, 3), 3 => 6),
    MaxPoolLayer((8, 8, 6), (2, 2)),
    ConvLayer((4, 4), (2, 2), 6 => 2),
    FlattenLayer((3, 3, 2)),
    DenseLayer(18 => 2)
)
@test size(model2) == ((10, 10, 3) => (2,))
@test size(model2(rand(10, 10, 3))) == (2,)

# test concatenation connection
model3 = Concat(
    Chain(
        DenseLayer(10 => 6),
        DenseLayer(6 => 3),
        DenseLayer(3 => 2)
    ),
    Chain(
        ConvLayer((10, 10), (5, 5), 1 => 3),
        MaxPoolLayer((6, 6, 3), (3, 3)),
        FlattenLayer(2, 2, 3),
        DenseLayer(12 => 10)
    )
)

@test size(model3) == (((10,), (10, 10, 1)) => (12,))
@test size(model3(rand(10), rand(10, 10, 1))) == (12,)

# test concatenation layer
model4 = Chain(
    Concat(
        Chain(
            ConvLayer((28, 28), (3, 3), 1 => 3),
            MaxPoolLayer((26, 26, 3), (2, 2)),
            ConvLayer((13, 13), (4, 4), 3 => 4),
            MaxPoolLayer((10, 10, 4), (5, 5)),
            FlattenLayer(2, 2, 4)
        ),
        IdentityLayer(10)
    ),
    DenseLayer(26 => 5),
    DenseLayer(5 => 1)
)

@test size(model4) == (((28, 28, 1), (10,)) => (1,))
@test size(model4(rand(28, 28, 1), rand(10))) == (1,)
@test length(diff(model4, [0.1])) == 2