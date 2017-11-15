using CUSPARSE
using CUDAdrv
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5
@testset "axpyi" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        @testset "axpyi!" begin
            x = sparsevec(rand(1:m,k), rand(elty,k), m)
            y = rand(elty,m)
            d_x = CudaSparseVector(x)
            d_y = CuArray(y)
            alpha = rand(elty)
            d_y = CUSPARSE.axpyi!(alpha,d_x,d_y,'O')
            #compare
            h_y = to_host(d_y)
            y[x.nzind] += alpha * x.nzval
            @test h_y ≈ y
        end

        @testset "axpyi" begin
            x = sparsevec(rand(1:m,k), rand(elty,k), m)
            y = rand(elty,m)
            d_x = CudaSparseVector(x)
            d_y = CuArray(y)
            alpha = rand(elty)
            d_z = CUSPARSE.axpyi(alpha,d_x,d_y,'O')
            #compare
            h_z = to_host(d_z)
            z = copy(y)
            z[x.nzind] += alpha * x.nzval
            @test h_z ≈ z
            d_z = CUSPARSE.axpyi(d_x,d_y,'O')
            #compare
            h_z = to_host(d_z)
            z = copy(y)
            z[x.nzind] += x.nzval
            @test h_z ≈ z
        end
    end
end
