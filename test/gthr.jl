using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

#############
# test_gthr #
#############

function test_gthr!(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrixCSC(x)
    d_y = CudaArray(y)
    d_y = CUSPARSE.gthr!(d_x,d_y,'O')
    h_x = to_host(d_x)
    x.nzval = y[x.rowval]
    @test_approx_eq(h_x,x)
end

function test_gthr(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrixCSC(x)
    d_y = CudaArray(y)
    d_z = CUSPARSE.gthr(d_x,d_y,'O')
    h_z = to_host(d_z)
    x.nzval = y[x.rowval]
    @test_approx_eq(h_z,x)
end

##############
# test_gthrz #
##############

function test_gthrz!(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrixCSC(x)
    d_y = CudaArray(y)
    d_x,d_y = CUSPARSE.gthrz!(d_x,d_y,'O')
    h_x = to_host(d_x)
    h_y = to_host(d_y)
    x.nzval = y[x.rowval]
    @test_approx_eq(h_x,x)
    y[x.rowval] = zero(elty)
    @test_approx_eq(h_y,y)
end

function test_gthrz(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrixCSC(x)
    d_y = CudaArray(y)
    d_z,d_w = CUSPARSE.gthrz(d_x,d_y,'O')
    h_w = to_host(d_w)
    h_z = to_host(d_z)
    x.nzval = y[x.rowval]
    @test_approx_eq(h_z,x)
    y[x.rowval] = zero(elty)
    @test_approx_eq(h_w,y)
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_gthr!(elty)
    println("gthr! took ", toq(), " for ", elty)
    tic()
    test_gthr(elty)
    println("gthr took ", toq(), " for ", elty)
    tic()
    test_gthrz!(elty)
    println("gthrz! took ", toq(), " for ", elty)
    tic()
    test_gthrz(elty)
    println("gthrz took ", toq(), " for ", elty)
end
