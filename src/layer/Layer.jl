abstract type AbstractLayerData end
abstract type AbstractLayer end

"All data types that can be accepted as input."
const NNDataType = Union{AbstractArray{Float64}, NTuple{N, Array{Float64}}} where N

(layer::AbstractLayer)(data::NNDataType) = predict(layer, data)
(layer::AbstractLayer)(data::NNDataType...) = predict(layer, Tuple(data))

include("IdentityLayer.jl")
include("ResizeLayer.jl")
include("DenseLayer.jl")
include("ConvLayer.jl")
include("MaxPoolLayer.jl")