using CUSPARSE
using CUDArt
using Base.Test

m = 5
n = 5
k = 10
blockdim = 5

#############
# test_geam #
#############

function test_geam_csr(elty)
    A = sparse(rand(elty,m,n))
    B = sparse(rand(elty,m,n))
    alpha = rand(elty)
    beta = rand(elty)
    C = alpha * A + beta * B
    d_A = CudaSparseMatrixCSR(A)
    d_B = CudaSparseMatrixCSR(B)
    d_C = CUSPARSE.geam(alpha,d_A,beta,d_B,'O','O','O')
    h_C = to_host(d_C)
    @test_approx_eq(C,h_C)
    d_C = CUSPARSE.geam(d_A,beta,d_B,'O','O','O')
    h_C = to_host(d_C)
    C = A + beta * B
    @test_approx_eq(C,h_C)
    d_C = CUSPARSE.geam(alpha,d_A,d_B,'O','O','O')
    h_C = to_host(d_C)
    C = alpha * A + B
    @test_approx_eq(C,h_C)
    d_C = CUSPARSE.geam(d_A,d_B,'O','O','O')
    h_C = to_host(d_C)
    C = A + B
    @test_approx_eq(C,h_C)
    B = sparse(rand(elty,k,n))
    d_B = CudaSparseMatrixCSR(B)
    @test_throws(DimensionMismatch,CUSPARSE.geam(d_B,d_A,'O','O','O'))
end

function test_geam_csc(elty)
    A = sparse(rand(elty,m,n))
    B = sparse(rand(elty,m,n))
    alpha = rand(elty)
    beta = rand(elty)
    C = alpha * A + beta * B
    d_A = CudaSparseMatrixCSC(A)
    d_B = CudaSparseMatrixCSC(B)
    d_C = CUSPARSE.geam(alpha,d_A,beta,d_B,'O','O','O')
    h_C = to_host(d_C)
    @test_approx_eq(C,h_C)
    d_C = CUSPARSE.geam(d_A,beta,d_B,'O','O','O')
    h_C = to_host(d_C)
    C = A + beta * B
    @test_approx_eq(C,h_C)
    d_C = CUSPARSE.geam(alpha,d_A,d_B,'O','O','O')
    h_C = to_host(d_C)
    C = alpha * A + B
    @test_approx_eq(C,h_C)
    d_C = CUSPARSE.geam(d_A,d_B,'O','O','O')
    h_C = to_host(d_C)
    C = A + B
    @test_approx_eq(C,h_C)
    B = sparse(rand(elty,k,n))
    d_B = CudaSparseMatrixCSC(B)
    @test_throws(DimensionMismatch,CUSPARSE.geam(d_B,d_A,'O','O','O'))
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_geam_csr(elty)
    test_geam_csc(elty)
    println("geam took ", toq(), " for ", elty)
end
