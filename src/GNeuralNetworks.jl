module GNeuralNetworks

import Base:size
import Base:diff
import Base:+
import Base:-
import Base:*
import Base:/

export onehot
export FunctionAndDerivative
export sigmoid, reLU, mae, cross_entropy
export AbstractLayer, IdentityLayer, ResizeLayer, FlattenLayer, DenseLayer, ConvLayer, MaxPoolLayer
export size, predict, diff, update!
export Chain, Concat

include("utils.jl")
include("function.jl")
include("layer/Layer.jl")
include("connection/Connection.jl")

end
