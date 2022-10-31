struct FunctionAndDerivative
    f::Function
    f_::Function
end

onehot(n, i, type=Float64) = [type(j == i) for j in 1:n]

sigmoid = FunctionAndDerivative(x -> 1 / (1 + exp(-x)), x -> exp(-x) / (1 + exp(-x)) ^ 2)
reLU = FunctionAndDerivative(x -> ((x > 0) ? x : 0), x -> ((x > 0) ? 1 : 0))

mae = FunctionAndDerivative(
    (y, ŷ) -> sum((y .- ŷ) .^ 2),
    (y, ŷ) -> 2 * (y - ŷ)
)
crossEntropy = FunctionAndDerivative(
    (y, ŷ) -> (-sum(ŷ .* log.(y)) - sum((1 .- ŷ) .* log.(1 .- ŷ))),
    (y, y̋) -> (-ŷ ./ y .+ (1 .- ŷ) / (1 .- y))
)