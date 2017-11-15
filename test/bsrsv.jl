using CUSPARSE
using CUDAdrv
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "bsrsv2" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        @testset "bsrsv2!" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CudaSparseMatrixCSR(sparse(A))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_X = CUSPARSE.sv2!('N','U',alpha,d_A,d_X,'O')
            h_Y = to_host(d_X)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_X = CuArray(rand(elty,n))
            @test_throws DimensionMismatch CUSPARSE.sv2!('N','U',alpha,d_A,d_X,'O')
            A = sparse(rand(elty,m,n))
            d_A = CudaSparseMatrixCSR(A)
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            @test_throws DimensionMismatch CUSPARSE.sv2!('N','U',alpha,d_A,d_X,'O')
        end

        @testset "bsrsv2" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CudaSparseMatrixCSR(sparse(A))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_Y = CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
            h_Y = to_host(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_Y = UpperTriangular(d_A)\d_X
            h_Y = to_host(d_Y)
            @test h_Y ≈ A\X
            A = sparse(rand(elty,m,n))
            d_A = CudaSparseMatrixCSR(A)
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            @test_throws DimensionMismatch CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
        end
    end
end
