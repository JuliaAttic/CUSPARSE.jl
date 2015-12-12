using CUSPARSE
using CUDArt
using Base.Test

m = 5
n = 35
k = 10
blockdim = 5

##############
# test_cscsv #
##############

function test_cscsv(elty)
    A = rand(elty,m,m)
    A = triu(A)
    x = rand(elty,m)
    alpha = rand(elty)
    d_x = CudaArray(x)
    d_A = CudaSparseMatrixCSC(sparse(A))
    d_y = CUSPARSE.sv('N','T','U',alpha,d_A,d_x,'O')
    h_y = to_host(d_y)
    y = A\(alpha * x)
    @test_approx_eq(y,h_y)
    x = rand(elty,n)
    d_x = CudaArray(x)
    info = CUSPARSE.sv_analysis('N','T','U',d_A,'O')
    @test_throws(DimensionMismatch, CUSPARSE.sv_solve('N','U',alpha,d_A,d_x,info,'O'))
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSC(A)
    @test_throws(DimensionMismatch, CUSPARSE.sv_analysis('T','T','U',d_A,'O'))
    CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
end

###############
# test_sv2 #
###############

function test_cscsv2!(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSC(sparse(A))
    d_X = CUSPARSE.sv2!('N','U',alpha,d_A,d_X,'O')
    h_Y = to_host(d_X)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    d_X = CudaArray(rand(elty,n))
    @test_throws(DimensionMismatch, CUSPARSE.sv2!('N','U',alpha,d_A,d_X,'O'))
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSC(sparse(A))
    @test_throws(DimensionMismatch, CUSPARSE.sv2!('N','U',alpha,d_A,d_X,'O'))
end

function test_cscsv2(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSC(sparse(A))
    d_Y = CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
    h_Y = to_host(d_Y)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSC(A)
    @test_throws(DimensionMismatch, CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O'))
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_cscsv(elty)
    println("cscsv took ", toq(), " for ", elty)
    tic()
    test_cscsv2!(elty)
    println("sv2! took ", toq(), " for ", elty)
    tic()
    test_cscsv2(elty)
    println("sv2 took ", toq(), " for ", elty)
end
