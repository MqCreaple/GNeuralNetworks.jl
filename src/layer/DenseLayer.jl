mutable struct DenseLayerData <: AbstractLayerData
    weight::Matrix{Float64}
    bias::Vector{Float64}
end

+(d1::DenseLayerData, d2::DenseLayerData) = DenseLayerData(d1.weight + d2.weight, d1.bias + d2.bias)
-(d1::DenseLayerData, d2::DenseLayerData) = DenseLayerData(d1.weight - d2.weight, d1.bias - d2.bias)
*(d1::DenseLayerData, scalar::Float64) = DenseLayerData(d1.weight * scalar, d1.bias * scalar)
/(d1::DenseLayerData, scalar::Float64) = DenseLayerData(d1.weight / scalar, d1.bias / scalar)

"""
    Dense(inLen => outLen; activation = sigmoid)

Dense layer. Edges connect any neuron in last layer to any neuron in next layer.

`inLen` is the length of input vector, `outLen` is the length of output vector.

Dense layer only accepts one-dimensional vectors as input.
"""
mutable struct DenseLayer <: AbstractLayer
    # layer data
    data::DenseLayerData
    activation::FunctionAndDerivative
    # input and output data
    lastInput::Vector{Float64}
    lastOutputBefore::Vector{Float64}
    # constructor
    DenseLayer(pair::Pair{Int, Int}; activation::FunctionAndDerivative=sigmoid) = new(
        DenseLayerData(randn(pair.second, pair.first), randn(pair.second)),
        activation,
        zeros(pair.first),
        zeros(pair.second)
    )
end

"dimension of dense layer"
function size(layer::DenseLayer)
    a, b = size(layer.data.weight)
    return (b,) => (a,)
end

function predict(layer::DenseLayer, data::Vector{Float64})
    layer.lastInput = data
    layer.lastOutputBefore = layer.data.weight * data + layer.data.bias
    return layer.activation.f.(layer.lastOutputBefore)
end

function diff(layer::DenseLayer, outputDiff::Vector{Float64})
    biasDiff = outputDiff .* layer.activation.f_.(layer.lastOutputBefore)
    weightDiff = biasDiff * transpose(layer.lastInput)
    layerDiff = transpose(layer.data.weight) * biasDiff
    return DenseLayerData(weightDiff, biasDiff), layerDiff
end

# TODO necessary?
function update!(layer::DenseLayer, data::DenseLayerData)
    layer.data -= data
end