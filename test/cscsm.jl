using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

##############
# test_cscsm #
##############

function test_cscsm(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m,n)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSC(sparse(A))
    info = CUSPARSE.cscsm_analysis('N','U',d_A,'O')
    d_Y = CUSPARSE.cscsm_solve('N','U',alpha,d_A,d_X,info,'O')
    h_Y = to_host(d_Y)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    d_X = CudaArray(rand(elty,n,n))
    @test_throws(DimensionMismatch, CUSPARSE.cscsm_solve('N','U',alpha,d_A,d_X,info,'O'))
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSC(A)
    @test_throws(DimensionMismatch, CUSPARSE.cscsm_analysis('T','U',d_A,'O'))
    CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_cscsm(elty)
    println("cscsm took ", toq(), " for ", elty)
end
