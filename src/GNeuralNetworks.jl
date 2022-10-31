module GNeuralNetworks

import Base:size
import Base:+
import Base:-
import Base:*
import Base:/

export onehot
export FunctionAndDerivative
export sigmoid, reLU, mae, crossEntropy
export AbstractLayer, ResizeLayer, DenseLayer, ConvLayer, MaxPoolLayer
export Model

include("function.jl")
include("layer/layer.jl")
include("Model.jl")

end
