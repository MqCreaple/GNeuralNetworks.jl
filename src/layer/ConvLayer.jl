mutable struct ConvLayerData <: AbstractLayerData
    core::Array{Float64, 4}
end

+(d1::ConvLayerData, d2::ConvLayerData) = ConvLayerData(d1.core + d2.core)
-(d1::ConvLayerData, d2::ConvLayerData) = ConvLayerData(d1.core - d2.core)
*(d1::ConvLayerData, scalar::Float64) = ConvLayerData(d1.core * scalar)
/(d1::ConvLayerData, scalar::Float64) = ConvLayerData(d1.core / scalar)

"""
    ConvLayer((imageX, imageY), (coreX, coreY), inDim => outDim)

Convolution Layer.

`imageX` and `imageY` are the width and height of input images; `coreX` and `coreY` are the width and height of convolution core.

`inDim` is number of channels in input, `outDim` is number of channels in output.

The convolution layer accepts a rank-3 tensor with dimension `(imageX, imageY, inDim)` and outputs a rank-3 tensor with dimension `(imageX-coreX+1, imageY-coreY+1, outDIm)`.
"""
mutable struct ConvLayer <: AbstractLayer
    data::ConvLayerData
    imageSize::Tuple{Int, Int}
    # last input
    lastInput::Array{Float64, 3}
    ConvLayer(imageSize::Tuple{Int, Int}, core::Tuple{Int, Int}, ioChannels::Pair{Int, Int}) = new(
        ConvLayerData(randn(core[1], core[2], ioChannels.first, ioChannels.second)),
        imageSize,
        zeros(imageSize[1], imageSize[2], ioChannels[1])
    )
end

"dimension of convolution layer"
function size(layer::ConvLayer)
    szCore = size(layer.data.core)
    return (layer.imageSize[1], layer.imageSize[2], szCore[3]) =>
            (layer.imageSize[1] - szCore[1] + 1, layer.imageSize[2] - szCore[2] + 1, szCore[4])
end

function predict(layer::ConvLayer, data::Array{Float64, 3})
    layer.lastInput = data
    sz = size(layer)
    ans = zeros(sz.second)
    @inbounds for l in 1:sz.second[3]
        for j in 1:sz.second[2]
            for i in 1:sz.second[1]
                # calculate convolution on answer's pixel (i, j) and channel l
                xInd = i:i+size(layer.data.core)[1]-1
                yInd = j:j+size(layer.data.core)[2]-1
                ans[i, j, l] = sum(view(layer.data.core, :, :, :, l) .* view(data, xInd, yInd, :))
            end
        end
    end
    return ans
end

function diff(layer::ConvLayer, outputDiff::Array{Float64, 3})
    coreDiff = zeros(size(layer.data.core))
    inputDiff = zeros(size(layer).first)
    # partial derivatives wrt. convolution core
    @inbounds for m in 1:size(coreDiff)[4]
        for l in 1:size(coreDiff)[3]
            for j in 1:size(coreDiff)[2]
                for i in 1:size(coreDiff)[1]
                    xInd = i:i+size(outputDiff)[1]-1
                    yInd = j:j+size(outputDiff)[2]-1
                    coreDiff[i, j, l, m] = sum(view(outputDiff, :, :, m) .* view(layer.lastInput, xInd, yInd, l))
                end
            end
        end
    end
    # partial derivatives wrt. input data
    @inbounds for l in 1:size(inputDiff)[3]
        for j in 1:size(inputDiff)[2]
            for i in 1:size(inputDiff)[1]
                xInd::UnitRange{Int} = i-size(layer.data.core)[1]+1:min(i, size(outputDiff)[1])
                yInd::UnitRange{Int} = j-size(layer.data.core)[2]+1:min(j, size(outputDiff)[2])
                cxInd::UnitRange{Int} = 1:length(xInd)
                cyInd::UnitRange{Int} = 1:length(yInd)
                if xInd.start <= 0
                    cxInd = 2-xInd.start:length(xInd)
                    xInd = 1:i
                end
                if yInd.start <= 0
                    cyInd = 2-yInd.start:length(yInd)
                    yInd = 1:j
                end
                inputDiff[i, j, l] = sum(
                    reverse(view(outputDiff, xInd, yInd, :), dims=(1, 2))
                    .* view(layer.data.core, cxInd, cyInd, l, :)
                )
            end
        end
    end
    return ConvLayerData(coreDiff), inputDiff
end

function update!(layer::ConvLayer, data::ConvLayerData)
    layer.data -= data
end