using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

##############
# test_csrsv #
##############

function test_csrsv(elty)
    A = rand(elty,m,m)
    A = triu(A)
    x = rand(elty,m)
    alpha = rand(elty)
    d_x = CudaArray(x)
    d_A = CudaSparseMatrixCSR(sparse(A))
    info = CUSPARSE.csrsv_analysis('N','T','U',d_A,'O')
    d_y = CUSPARSE.csrsv_solve('N','U',alpha,d_A,d_x,info,'O')
    h_y = to_host(d_y)
    y = A\(alpha * x)
    @test_approx_eq(y,h_y)
    x = rand(elty,n)
    d_x = CudaArray(x)
    @test_throws(DimensionMismatch, CUSPARSE.csrsv_solve('N','U',alpha,d_A,d_x,info,'O'))
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    @test_throws(DimensionMismatch, CUSPARSE.csrsv_analysis('T','T','U',d_A,'O'))
    CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
end

###############
# test_csrsv2 #
###############

function test_csrsv2!(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_X = CUSPARSE.csrsv2!('N',alpha,d_A,d_X,'O')
    h_Y = to_host(d_X)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    d_X = CudaArray(rand(elty,n))
    @test_throws(DimensionMismatch, CUSPARSE.csrsv2!('N',alpha,d_A,d_X,'O'))
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(sparse(A))
    @test_throws(DimensionMismatch, CUSPARSE.csrsv2!('N',alpha,d_A,d_X,'O'))
end

function test_csrsv2(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_Y = CUSPARSE.csrsv2('N',alpha,d_A,d_X,'O')
    h_Y = to_host(d_Y)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    @test_throws(DimensionMismatch, CUSPARSE.csrsv2('N',alpha,d_A,d_X,'O'))
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_csrsv(elty)
    println("csrsv took ", toq(), " for ", elty)
    tic()
    test_csrsv2!(elty)
    println("csrsv2! took ", toq(), " for ", elty)
    tic()
    test_csrsv2(elty)
    println("csrsv2 took ", toq(), " for ", elty)
end
