abstract type AbstractLayerData end
abstract type AbstractLayer end

include("ResizeLayer.jl")
include("DenseLayer.jl")
include("ConvLayer.jl")
include("MaxPoolLayer.jl")