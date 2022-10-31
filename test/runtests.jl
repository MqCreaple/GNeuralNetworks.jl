using GNeuralNetworks
using Test

@testset "GNeuralNetworks.jl" begin
    @test onehot(5, 1) == [1.0, 0.0, 0.0, 0.0, 0.0]
    @test sigmoid.f(0) == 0.5
end
