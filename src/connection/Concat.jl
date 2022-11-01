"""
    Concat(layer1, layer2, ...)

Concatenation layer.

This layer contains `n` sub-layers, which would accept `n` inputs, calculate them in parallel, and concatenate the result to one vector.

The output of each sub-layer must be a vector (1-dimensional array). If not, consider adding a flatten layer at the end.
"""
mutable struct Concat{N} <: AbstractLayer
    layers::NTuple{N, AbstractLayer}
    prefSum::Vector{Int}
    Concat(layers::NTuple{N, AbstractLayer}) where N = begin
        @assert all(length.(last.(size.(layers))) .== 1)  # check if all layers's output data are vectors
        pSum = collect(first.(last.(size.(layers))))      # calculate prefix sum of the output dimensions
        new{N}(layers, prefix_sum!(pSum))
    end
end

Concat(layers::AbstractLayer...) = Concat(Tuple(layers))

size(layer::Concat{N}) where N = (first.(size.(layer.layers)) => (layer.prefSum[end],))

function predict(layer::Concat{N}, data::NTuple{N, NNDataType}) where N
    return cat(ntuple(i -> predict(layer.layers[i], data[i]), N)...; dims=1)
end

function diff(layer::Concat{N}, outputDiff::Vector{Float64}) where N
    # TODO (use array slices to increase performance)
    diffs = [diff(layer.layers[i], outputDiff[((i == 1) ? 1 : layer.prefSum[i-1] + 1):layer.prefSum[i]])
        for i in 1:N]
    return first.(diffs), last.(diffs)
end

function update!(layer::Concat{N}, data::Vector) where N
    for i in eachindex(layer.layers)
        update!(layer.layers[i], data[i])
    end
end