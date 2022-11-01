using GNeuralNetworks
using Test

@testset "GNeuralNetworks.jl" begin
    @testset "function" begin
        include("function.jl")
    end
    @testset "connection" begin
        include("connection.jl")
    end
end