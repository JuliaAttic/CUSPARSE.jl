using CUSPARSE
using CUDAdrv
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "gthr and gthrz" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        x = sparsevec(rand(1:m,k), rand(elty,k), m)
        y = rand(elty,m)
        @testset "gthr!" begin
            d_x = CudaSparseVector(x)
            d_y = CuArray(y)
            d_y = CUSPARSE.gthr!(d_x,d_y,'O')
            h_x = to_host(d_x)
            @test h_x ≈ SparseVector(m,x.nzind,y[x.nzind])
        end

        @testset "gthr" begin
            d_x = CudaSparseVector(x)
            d_y = CuArray(y)
            d_z = CUSPARSE.gthr(d_x,d_y,'O')
            h_z = to_host(d_z)
            @test h_z ≈ SparseVector(m,x.nzind,y[x.nzind])
        end

        @testset "gthrz!" begin
            d_x = CudaSparseVector(x)
            d_y = CuArray(y)
            d_x,d_y = CUSPARSE.gthrz!(d_x,d_y,'O')
            h_x = to_host(d_x)
            h_y = to_host(d_y)
            @test h_x ≈ SparseVector(m,x.nzind,y[x.nzind])
            y[x.nzind] = zero(elty)
            @test h_y ≈ y
        end

        @testset "gthrz" begin
            d_x = CudaSparseVector(x)
            d_y = CuArray(y)
            d_z,d_w = CUSPARSE.gthrz(d_x,d_y,'O')
            h_w = to_host(d_w)
            h_z = to_host(d_z)
            @test h_z ≈ SparseVector(m,x.nzind,y[x.nzind])
            y[x.nzind] = zero(elty)
            @test h_w ≈ y
        end
    end
end
