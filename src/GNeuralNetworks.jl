module GNeuralNetworks

import Base:size
import Base:diff
import Base:+
import Base:-
import Base:*
import Base:/

export onehot
export FunctionAndDerivative
export sigmoid, reLU, mae, crossEntropy
export AbstractLayer, ResizeLayer, flatten, DenseLayer, ConvLayer, MaxPoolLayer
export size, predict, diff, update!
export Model

include("function.jl")
include("layer/layer.jl")
include("Model.jl")

end
