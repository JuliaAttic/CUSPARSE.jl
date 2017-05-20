using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "sctr" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        x = sparsevec(rand(1:m,k), rand(elty,k), m)
        y = zeros(elty,m)
        @testset "sctr!" begin
            d_x = CudaSparseVector(x)
            d_y = CudaArray(y)
            d_y = CUSPARSE.sctr!(d_x,d_y,'O')
            h_y = to_host(d_y)
            y[x.nzind]  += x.nzval
            @test h_y ≈ y
        end
        y = zeros(elty,m)

        @testset "sctr" begin
            d_x = CudaSparseVector(x)
            d_y = CUSPARSE.sctr(d_x,'O')
            h_y = to_host(d_y)
            y = zeros(elty,m)
            y[x.nzind]  += x.nzval
            @test h_y ≈ y
        end
    end
end
