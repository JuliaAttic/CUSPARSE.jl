using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

##############
# test_bsrmm #
##############

function test_bsrmm!(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
    d_C = CUSPARSE.bsrmm!('N','N',alpha,d_A,d_B,beta,d_C,'O')
    h_C = to_host(d_C)
    C = alpha * A * B + beta * C
    @test_approx_eq(C,h_C)
    @test_throws(DimensionMismatch, CUSPARSE.bsrmm!('N','T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.bsrmm!('T','N',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.bsrmm!('T','T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.bsrmm!('N','N',alpha,d_A,d_B,beta,d_B,'O'))
end

function test_bsrmm(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
    d_D = CUSPARSE.bsrmm('N','N',alpha,d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = alpha * A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.bsrmm('N','N',d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.bsrmm('N','N',d_A,d_B,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.bsrmm('N','N',alpha,d_A,d_B,'O')
    h_D = to_host(d_D)
    D = alpha * A * B
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.bsrmm('N','N',d_A,d_B,'O')
    h_D = to_host(d_D)
    D = A * B
    @test_approx_eq(D,h_D)
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_bsrmm!(elty)
    println("bsrmm! took ", toq(), " for ", elty)
    tic()
    test_bsrmm(elty)
    println("bsrmm took ", toq(), " for ", elty)
end
