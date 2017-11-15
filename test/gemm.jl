using CUSPARSE
using CUDAdrv
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "gemm" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        A = sparse(rand(elty,m,k))
        B = sparse(rand(elty,k,n))
        C = A * B
        d_A = CudaSparseMatrixCSR(A)
        d_B = CudaSparseMatrixCSR(B)
        d_C = CUSPARSE.gemm('N','N',d_A,d_B,'O','O','O')
        r_r = to_host(d_C.rowPtr)
        r_c = to_host(d_C.colVal)
        r_v = to_host(d_C.nzVal)
        h_C = to_host(d_C)
        @test C ≈ h_C
        @test_throws DimensionMismatch CUSPARSE.gemm('N','T',d_A,d_B,'O','O','O')
        @test_throws DimensionMismatch CUSPARSE.gemm('T','T',d_A,d_B,'O','O','O')
        @test_throws DimensionMismatch CUSPARSE.gemm('T','N',d_A,d_B,'O','O','O')
        @test_throws DimensionMismatch CUSPARSE.gemm('N','N',d_B,d_A,'O','O','O')
        #=A = sparse(rand(elty,m,k))
        B = sparse(rand(elty,k,n))
        d_A = CudaSparseMatrixCSC(A)
        d_B = CudaSparseMatrixCSC(B)
        C = A * B
        d_C = CUSPARSE.gemm('N','N',d_A,d_B,'O','O','O')
        h_C = to_host(d_C)
        @test_approx_eq(C,h_C)
        @test_throws(DimensionMismatch,CUSPARSE.gemm('N','T',d_A,d_B,'O','O','O'))
        @test_throws(DimensionMismatch,CUSPARSE.gemm('T','T',d_A,d_B,'O','O','O'))
        @test_throws(DimensionMismatch,CUSPARSE.gemm('T','N',d_A,d_B,'O','O','O'))
        @test_throws(DimensionMismatch,CUSPARSE.gemm('N','N',d_B,d_A,'O','O','O'))=#
    end
end
