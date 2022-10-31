"""
    MaxPoolLayer((imageX, imageY, inDim), (windowX, windowY); activation = sigmoid)

Maximum pooling layer.
"""
struct MaxPoolLayer <: AbstractLayer
    activation::FunctionAndDerivative
    inputSize::Tuple{Int, Int, Int}
    windowSize::Tuple{Int, Int}
    # last input data
    lastMax::Array{Float64, 3}                # the maximum value of each cell last time
    lastMaxIndex::Array{CartesianIndex{2}, 3} # the index of last maximum value last time
    MaxPoolLayer(inputSize::Tuple{Int, Int, Int}, windowSize::Tuple{Int, Int}; activation=sigmoid) = new(
        activation, inputSize, windowSize,
        fill(0.0, (cld(inputSize[1], windowSize[1]), cld(inputSize[2], windowSize[2]), inputSize[3])),
        fill(CartesianIndex(0, 0), (cld(inputSize[1], windowSize[1]), cld(inputSize[2], windowSize[2]), inputSize[3]))
    )
end

"dimension of maximum pooling layer"
size(layer::MaxPoolLayer) = (layer.inputSize => size(layer.lastMax))

function predict(layer::MaxPoolLayer, data::Array{Float64, 3})
    sz = size(layer.lastMax)
    @inbounds for l in 1:sz[3]
        for j in 1:sz[2]
            for i in 1:sz[1]
                xBound = ((i-1) * layer.windowSize[1] + 1) : min(i * layer.windowSize[1], layer.inputSize[1])
                yBound = ((j-1) * layer.windowSize[2] + 1) : min(j * layer.windowSize[2], layer.inputSize[2])
                layer.lastMax[i, j, l], layer.lastMaxIndex[i, j, l] = findmax(view(data, xBound, yBound, l))
            end
        end
    end
    return layer.activation.f.(layer.lastMax)
end

function diff(layer::MaxPoolLayer, outputDiff::Array{Float64, 3})
    ans = zeros(layer.inputSize)
    sz = size(layer.lastMax)
    @inbounds for l in 1:sz[3]
        for j in 1:sz[2]
            for i in 1:sz[1]
                xInd = (i-1) * layer.windowSize[1] + layer.lastMaxIndex[i, j, l][1]
                yInd = (j-1) * layer.windowSize[2] + layer.lastMaxIndex[i, j, l][2]
                ans[xInd, yInd, l] = outputDiff[i, j, l] * layer.activation.f_.(layer.lastMax[i, j, l])
            end
        end
    end
    return missing, ans
end

function update!(layer::MaxPoolLayer, data) end