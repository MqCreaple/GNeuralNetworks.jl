"""
    ResizeLayer(inputSize, outputSize)

Resize layer. The layer takes an array and resize it to another shape.
"""
struct ResizeLayer <: AbstractLayer
    inputSize::Tuple{Vararg{Int}}
    outputSize::Tuple{Vararg{Int}}
    ResizeLayer(inputSize::Tuple{Vararg{Int}}, outputSize::Tuple{Vararg{Int}}) = begin
        @assert prod(inputSize) == prod(outputSize)
        new(inputSize, outputSize)
    end
end

"dimension of resize layer"
size(layer::ResizeLayer) = (layer.inputSize => layer.outputSize)

function predict(layer::ResizeLayer, data::Array{Float64})
    return reshape(data, layer.outputSize)
end

function diff(layer::ResizeLayer, outputDiff::Array{Float64})
    return missing, reshape(outputDiff, layer.inputSize)
end

function update!(layer::ResizeLayer, data) end

"a special kind of resize layer"
flatten(inputSize::Tuple{Vararg{Int}}) = ResizeLayer(inputSize, (prod(inputSize),))