using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

###############
# test_bsrsm2 #
###############

function test_bsrsm2!(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m,n)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    d_X = CUSPARSE.bsrsm2!('N','N',alpha,d_A,d_X,'O')
    h_Y = to_host(d_X)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    @test_throws(DimensionMismatch, CUSPARSE.bsrsm2!('N','N',alpha,d_A,d_X,'O'))
end

function test_bsrsm2(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m,n)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    d_Y = CUSPARSE.bsrsm2('N','N',alpha,d_A,d_X,'O')
    h_Y = to_host(d_Y)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    @test_throws(DimensionMismatch, CUSPARSE.bsrsm2('N','N',alpha,d_A,d_X,'O'))
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_bsrsm2!(elty)
    println("bsrsm2! took ", toq(), " for ", elty)
    tic()
    test_bsrsm2(elty)
    println("bsrsm2 took ", toq(), " for ", elty)
end
