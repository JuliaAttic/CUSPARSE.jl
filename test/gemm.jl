using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

#############
# test_gemm #
#############

function test_gemm(elty)
    A = sparse(rand(elty,m,k))
    B = sparse(rand(elty,k,n))
    C = A * B
    d_A = CudaSparseMatrixCSR(A)
    d_B = CudaSparseMatrixCSR(B)
    d_C = CUSPARSE.gemm('N','N',d_A,d_B,'O','O','O')
    h_C = to_host(d_C)
    @test_approx_eq(C,h_C)
    @test_throws(DimensionMismatch,CUSPARSE.gemm('N','T',d_A,d_B,'O','O','O'))
    @test_throws(DimensionMismatch,CUSPARSE.gemm('T','T',d_A,d_B,'O','O','O'))
    @test_throws(DimensionMismatch,CUSPARSE.gemm('T','N',d_A,d_B,'O','O','O'))
    @test_throws(DimensionMismatch,CUSPARSE.gemm('N','N',d_B,d_A,'O','O','O'))
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_gemm(elty)
    println("gemm took ", toq(), " for ", elty)
end
