using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

##############
# test_hybsv #
##############

function test_hybsv(elty)
    A = rand(elty,m,m)
    A = triu(A)
    x = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_A = CUSPARSE.switch2hyb(d_A)
    info = CUSPARSE.hybsv_analysis('N','U',d_A,'O')
    d_y = CUSPARSE.hybsv_solve('N','U',alpha,d_A,d_x,info,'O')
    h_y = to_host(d_y)
    y = A\(alpha * x)
    @test_approx_eq(y,h_y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2hyb(d_A)
    @test_throws(DimensionMismatch, CUSPARSE.hybsv_analysis('T','U',d_A,'O'))
    CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_hybsv(elty)
    println("hybsv took ", toq(), " for ", elty)
end
