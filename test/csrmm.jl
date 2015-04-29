using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

##############
# test_csrmm #
##############

function test_csrmm!(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_C = CUSPARSE.csrmm!('N',alpha,d_A,d_B,beta,d_C,'O')
    h_C = to_host(d_C)
    C = alpha * A * B + beta * C
    @test_approx_eq(C,h_C)
    @test_throws(DimensionMismatch, CUSPARSE.csrmm!('T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.csrmm!('N',alpha,d_A,d_B,beta,d_B,'O'))
end

function test_csrmm(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_D = CUSPARSE.csrmm('N',alpha,d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = alpha * A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm('N',d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm('N',d_A,d_B,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm('N',alpha,d_A,d_B,'O')
    h_D = to_host(d_D)
    D = alpha * A * B
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm('N',d_A,d_B,'O')
    h_D = to_host(d_D)
    D = A * B
    @test_approx_eq(D,h_D)
    @test_throws(DimensionMismatch, CUSPARSE.csrmm('T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.csrmm('N',alpha,d_A,d_B,beta,d_B,'O'))
end

###############
# test_csrmm2 #
###############

function test_csrmm2!(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_C = CUSPARSE.csrmm2!('N','N',alpha,d_A,d_B,beta,d_C,'O')
    h_C = to_host(d_C)
    C = alpha * A * B + beta * C
    @test_approx_eq(C,h_C)
    @test_throws(DimensionMismatch, CUSPARSE.csrmm2!('N','T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.csrmm2!('T','N',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.csrmm2!('T','T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.csrmm2!('N','N',alpha,d_A,d_B,beta,d_B,'O'))
end

function test_csrmm2(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_D = CUSPARSE.csrmm2('N','N',alpha,d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = alpha * A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm2('N','N',d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm2('N','N',d_A,d_B,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm2('N','N',alpha,d_A,d_B,'O')
    h_D = to_host(d_D)
    D = alpha * A * B
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm2('N','N',d_A,d_B,'O')
    h_D = to_host(d_D)
    D = A * B
    @test_approx_eq(D,h_D)
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_csrmm!(elty)
    println("csrmm! took ", toq(), " for ", elty)
    tic()
    test_csrmm(elty)
    println("csrmm took ", toq(), " for ", elty)
    tic()
    test_csrmm2!(elty)
    println("csrmm2! took ", toq(), " for ", elty)
    tic()
    test_csrmm2(elty)
    println("csrmm2 took ", toq(), " for ", elty)
end
