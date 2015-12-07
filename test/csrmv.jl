using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

##############
# test_mv #
##############

function test_mv!(elty)
    A = sparse(rand(elty,m,n))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    d_A = CudaSparseMatrixCSR(A)
    d_y = CUSPARSE.mv!('N',alpha,d_A,d_x,beta,d_y,'O')
    h_y = to_host(d_y)
    y = alpha * A * x + beta * y
    @test_approx_eq(y,h_y)
    @test_throws(DimensionMismatch, CUSPARSE.mv!('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.mv!('N',alpha,d_A,d_y,beta,d_x,'O'))
end

function test_mv_symm!(elty)
    A = sparse(rand(elty,m,m))
    A = A + A.'
    x = rand(elty,m)
    y = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    d_A = Symmetric(CudaSparseMatrixCSR(A))
    d_y = CUSPARSE.mv!('N',alpha,d_A,d_x,beta,d_y,'O')
    h_y = to_host(d_y)
    y = alpha * A * x + beta * y
    @test_approx_eq(y,h_y)
    x   = rand(elty,n)
    d_x = CudaArray(x)
    @test_throws(DimensionMismatch, CUSPARSE.mv!('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.mv!('N',alpha,d_A,d_y,beta,d_x,'O'))
end

function test_mv_herm!(elty)
    A = sparse(rand(elty,m,m))
    A = A + A'
    x = rand(elty,m)
    y = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    d_A = Hermitian(CudaSparseMatrixCSR(A))
    d_y = CUSPARSE.mv!('N',alpha,d_A,d_x,beta,d_y,'O')
    h_y = to_host(d_y)
    y = alpha * A * x + beta * y
    @test_approx_eq(y,h_y)
    x   = rand(elty,n)
    d_x = CudaArray(x)
    @test_throws(DimensionMismatch, CUSPARSE.mv!('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.mv!('N',alpha,d_A,d_y,beta,d_x,'O'))
end

function test_mv(elty)
    A = sparse(rand(elty,m,n))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    d_A = CudaSparseMatrixCSR(A)
    d_z = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',alpha,d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',alpha,d_A,d_x,'O')
    h_z = to_host(d_z)
    z = alpha * A * x
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',d_A,d_x,'O')
    h_z = to_host(d_z)
    z = A * x
    @test_approx_eq(z,h_z)
    @test_throws(DimensionMismatch, CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O'))
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_mv!(elty)
    println("mv! took ", toq(), " for ", elty)
    tic()
    test_mv_symm!(elty)
    println("mv_symm! took ", toq(), " for ", elty)
    tic()
    test_mv_herm!(elty)
    println("mv_herm! took ", toq(), " for ", elty)
    tic()
    test_mv(elty)
    println("mv took ", toq(), " for ", elty)
end
