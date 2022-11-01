"""
    Chain(layers...)

Chain connection. Layers are connected successively, with the output of last layer as the input of next layer.
"""
struct Chain <: AbstractLayer
    layers::Vector{AbstractLayer}
    Chain(args...) = begin
        for i in 2:lastindex(args)
            @assert size(args[i]).first == size(args[i-1]).second
        end
        new(collect(args))
    end
end

size(layer::Chain) = (size(layer.layers[begin]).first => size(layer.layers[end]).second)

function predict(layer::Chain, data::NNDataType)
    ans = data
    for i in eachindex(layer.layers)
        ans = predict(layer.layers[i], ans)
    end
    return ans
end

function diff(layer::Chain, outputDiff::NNDataType)
    layerDiffs::Vector = []
    for i in reverse(eachindex(layer.layers))
        curLayerDiff, outputDiff = diff(layer.layers[i], outputDiff)
        pushfirst!(layerDiffs, curLayerDiff)
    end
    return layerDiffs, outputDiff
end

function update!(layer::Chain, layerDiffs::Vector)
    for i in eachindex(layer.layers)
        @inbounds update!(layer.layers[i], layerDiffs[i])
    end
end