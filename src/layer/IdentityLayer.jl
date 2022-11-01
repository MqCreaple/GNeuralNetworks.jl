"""
    IdentityLayer(inSize)
    IdentityLayer(in1, in2, ...)

Identity layer. The layer will always return the exact data given as input.
"""
struct IdentityLayer{N} <: AbstractLayer
    inputSize::NTuple{N, Int}
end

IdentityLayer(args...) = IdentityLayer{length(args)}(Tuple(args))

size(layer::IdentityLayer) = (layer.inputSize => layer.inputSize)

predict(layer::IdentityLayer{N}, data::Array{Float64, N}) where N = data

diff(layer::IdentityLayer{N}, outputDiff::Array{Float64, N}) where N = outputDiff

function update!(layer::IdentityLayer, data) end