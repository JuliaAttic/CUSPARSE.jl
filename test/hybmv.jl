using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

##############
# test_hybmv #
##############

function test_hybmv!(elty)
    A = sparse(rand(elty,m,n))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2hyb(d_A)
    d_y = CUSPARSE.hybmv!('N',alpha,d_A,d_x,beta,d_y,'O')
    h_y = to_host(d_y)
    y = alpha * A * x + beta * y
    @test_approx_eq(y,h_y)
    @test_throws(DimensionMismatch, CUSPARSE.hybmv!('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.hybmv!('N',alpha,d_A,d_y,beta,d_x,'O'))
end

function test_hybmv(elty)
    A = sparse(rand(elty,m,n))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2hyb(d_A)
    d_z = CUSPARSE.hybmv('N',alpha,d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.hybmv('N',d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.hybmv('N',d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.hybmv('N',alpha,d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.hybmv('N',alpha,d_A,d_x,'O')
    h_z = to_host(d_z)
    z = alpha * A * x
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.hybmv('N',d_A,d_x,'O')
    h_z = to_host(d_z)
    z = A * x
    @test_approx_eq(z,h_z)
    @test_throws(DimensionMismatch, CUSPARSE.hybmv('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.hybmv('N',alpha,d_A,d_y,beta,d_x,'O'))
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_hybmv!(elty)
    println("hybmv! took ", toq(), " for ", elty)
    tic()
    test_hybmv(elty)
    println("hybmv took ", toq(), " for ", elty)
end
