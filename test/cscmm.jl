using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

##############
# test_cscmm #
##############

function test_cscmm!(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSC(A)
    d_C = CUSPARSE.cscmm!('N',alpha,d_A,d_B,beta,d_C,'O')
    h_C = to_host(d_C)
    C = alpha * A * B + beta * C
    @test_approx_eq(C,h_C)
    @test_throws(DimensionMismatch, CUSPARSE.cscmm!('T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.cscmm!('N',alpha,d_A,d_B,beta,d_B,'O'))
end

function test_cscmm(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSC(A)
    d_D = CUSPARSE.cscmm('N',alpha,d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = alpha * A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.cscmm('N',d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.cscmm('N',d_A,d_B,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.cscmm('N',alpha,d_A,d_B,'O')
    h_D = to_host(d_D)
    D = alpha * A * B
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.cscmm('N',d_A,d_B,'O')
    h_D = to_host(d_D)
    D = A * B
    @test_approx_eq(D,h_D)
    @test_throws(DimensionMismatch, CUSPARSE.cscmm('T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.cscmm('N',alpha,d_A,d_B,beta,d_B,'O'))
end

###############
# test_cscmm2 #
###############

function test_cscmm2!(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSC(A)
    d_C = CUSPARSE.cscmm2!('N','N',alpha,d_A,d_B,beta,d_C,'O')
    h_C = to_host(d_C)
    C = alpha * A * B + beta * C
    @test_approx_eq(C,h_C)
    @test_throws(DimensionMismatch, CUSPARSE.cscmm2!('N','T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.cscmm2!('T','N',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.cscmm2!('T','T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.cscmm2!('N','N',alpha,d_A,d_B,beta,d_B,'O'))
end

function test_cscmm2(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSC(A)
    d_D = CUSPARSE.cscmm2('N','N',alpha,d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = alpha * A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.cscmm2('N','N',d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.cscmm2('N','N',d_A,d_B,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.cscmm2('N','N',alpha,d_A,d_B,'O')
    h_D = to_host(d_D)
    D = alpha * A * B
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.cscmm2('N','N',d_A,d_B,'O')
    h_D = to_host(d_D)
    D = A * B
    @test_approx_eq(D,h_D)
end

types = [Float32,Float64]
for elty in types
    tic()
    test_cscmm!(elty)
    println("cscmm! took ", toq(), " for ", elty)
    tic()
    test_cscmm(elty)
    println("cscmm took ", toq(), " for ", elty)
    tic()
    test_cscmm2!(elty)
    println("cscmm2! took ", toq(), " for ", elty)
    tic()
    test_cscmm2(elty)
    println("cscmm2 took ", toq(), " for ", elty)
end
