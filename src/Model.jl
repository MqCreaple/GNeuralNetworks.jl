"""
Chain model.
"""
struct Model
    layers::Vector{AbstractLayer}
    η::Float64                  # learning rate
    loss::FunctionAndDerivative
    Model(args...; loss=mae, kwargs...) = begin
        for i in 2:length(args)
            @assert size(args[i-1]).second == size(args[i]).first
        end
        new(collect(args), kwargs[:η], loss)
    end
end

"Get the dimension of a model"
function size(model::Model)
    return size(model.layers[begin]).first => size(model.layers[end]).second
end

function predict(model::Model, data::Array{Float64})
    ans = data
    for i in 1:length(model.layers)
        ans = predict(model.layers[i], ans)
    end
    return ans
end

(model::Model)(data::Array{Float64}) = predict(model, data)

"""
Calculate partial derivatives of loss function with respect to parameters in model's each layer.

The input data and expected output data are given.
"""
function diff(model::Model, inputData::Array{Float64}, expected::Array{Float64})
    outDiff = model.loss.f_(predict(model, inputData), expected)
    layerDiff::Vector{Union{Missing, AbstractLayerData}} = []
    #// layerDiff::Vector{Any} = []
    for i in reverse(1:length(model.layers))
        curLayerDiff, outDiff = diff(model.layers[i], outDiff)
        pushfirst!(layerDiff, curLayerDiff)
    end
    return layerDiff
end

function train!(model::Model, inputData::Array{Float64}, expected::Array{Float64})
    layerDiffs = diff(model, inputData, expected)
    for i in 1:length(model.layers)
        update!(model.layers[i], layerDiffs[i] * model.η)
    end
end