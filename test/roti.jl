using CUSPARSE
using CuArrays
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "roti" begin
    @testset for elty in [Float32,Float64]
        x = sparsevec(rand(1:m,k), rand(elty,k), m)
        y = rand(elty,m)
        @testset "roti!" begin
            d_x = CudaSparseVector(x)
            d_y = CuArray(y)
            angle = rand(elty)
            d_x,d_y = CUSPARSE.roti!(d_x,d_y,cos(angle),sin(angle),'O')
            h_x = collect(d_x)
            h_y = collect(d_y)
            z = copy(x)
            w = copy(y)
            y[x.nzind] = cos(angle)*w[z.nzind] - sin(angle)*z.nzval
            @test h_x ≈ SparseVector(m,x.nzind,cos(angle)*z.nzval + sin(angle)*w[z.nzind])
            @test h_y ≈ y
        end

        @testset "roti" begin
            d_x = CudaSparseVector(x)
            d_y = CuArray(y)
            angle = rand(elty)
            d_z,d_w = CUSPARSE.roti(d_x,d_y,cos(angle),sin(angle),'O')
            h_w = collect(d_w)
            h_z = collect(d_z)
            z = copy(x)
            w = copy(y)
            w[z.nzind] = cos(angle)*y[x.nzind] - sin(angle)*x.nzval
            @test h_z ≈ SparseVector(m,z.nzind, cos(angle)*x.nzval + sin(angle)*y[x.nzind])
            @test h_w ≈ w
        end
    end
end
