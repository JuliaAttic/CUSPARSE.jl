using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "doti" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        x = sparsevec(rand(1:m,k), rand(elty,k), m)
        y = rand(elty,m)
        d_x = CudaSparseVector(x)
        d_y = CudaArray(y)
        ddot = CUSPARSE.doti(d_x,d_y,'O')
        #compare
        dot = zero(elty)
        for i in 1:length(x.nzval)
            dot += x.nzval[i] * y[x.nzind[i]]
        end
        @test ddot ≈ dot
    end
end

@testset "dotci" begin
    @testset for elty in [Complex64,Complex128]
        x = sparsevec(rand(1:m,k), rand(elty,k), m)
        y = rand(elty,m)
        d_x = CudaSparseVector(x)
        d_y = CudaArray(y)
        ddot = CUSPARSE.dotci(d_x,d_y,'O')
        #compare
        dot = zero(elty)
        for i in 1:length(x.nzval)
            dot += conj(x.nzval[i]) * y[x.nzind[i]]
        end
        @test ddot ≈ dot
    end
end
