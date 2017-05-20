using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "csrsm" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        @testset "csr" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m,n)
            alpha = rand(elty)
            d_X = CudaArray(X)
            d_A = CudaSparseMatrixCSR(sparse(A))
            info = CUSPARSE.sm_analysis('N','U',d_A,'O')
            d_Y = CUSPARSE.sm_solve('N','U',alpha,d_A,d_X,info,'O')
            h_Y = to_host(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_y = UpperTriangular(d_A)\d_X
            h_y = to_host(d_y)
            y = A\X
            @test y ≈ h_y
            d_X = CudaArray(rand(elty,n,n))
            @test_throws DimensionMismatch CUSPARSE.sm_solve('N','U',alpha,d_A,d_X,info,'O')
            A = sparse(rand(elty,m,n))
            d_A = CudaSparseMatrixCSR(A)
            @test_throws DimensionMismatch CUSPARSE.sm_analysis('T','U',d_A,'O')
            CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
        end
        @testset "csc" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m,n)
            alpha = rand(elty)
            d_X = CudaArray(X)
            d_A = CudaSparseMatrixCSC(sparse(A))
            info = CUSPARSE.sm_analysis('N','U',d_A,'O')
            d_Y = CUSPARSE.sm_solve('N','U',alpha,d_A,d_X,info,'O')
            h_Y = to_host(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_y = UpperTriangular(d_A)\d_X
            h_y = to_host(d_y)
            y = A\X
            @test y ≈ h_y
            d_X = CudaArray(rand(elty,n,n))
            @test_throws DimensionMismatch CUSPARSE.sm_solve('N','U',alpha,d_A,d_X,info,'O')
            A = sparse(rand(elty,m,n))
            d_A = CudaSparseMatrixCSC(A)
            @test_throws DimensionMismatch CUSPARSE.sm_analysis('T','U',d_A,'O')
            CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
        end
    end
end
