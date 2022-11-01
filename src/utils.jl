function prefix_sum!(arr::Vector{<:Number})
    for i in eachindex(arr)
        if i == firstindex(arr)
            continue
        end
        arr[i] += arr[i-1]
    end
    return arr
end