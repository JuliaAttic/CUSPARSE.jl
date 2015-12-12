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
    d_x = CudaSparseVector(x)
    d_y = CudaArray(y)
    d_y = CUSPARSE.gthr!(d_x,d_y,'O')
    h_x = to_host(d_x)
    nx = SparseVector(m,x.nzind,y[x.nzind])
    @test_approx_eq(h_x,nx)
end

function test_gthr(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseVector(x)
    d_y = CudaArray(y)
    d_z = CUSPARSE.gthr(d_x,d_y,'O')
    h_z = to_host(d_z)
    nx = SparseVector(m,x.nzind,y[x.nzind])
    @test_approx_eq(h_z,nx)
end

##############
# test_gthrz #
##############

function test_gthrz!(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseVector(x)
    d_y = CudaArray(y)
    d_x,d_y = CUSPARSE.gthrz!(d_x,d_y,'O')
    h_x = to_host(d_x)
    h_y = to_host(d_y)
    nx = SparseVector(m,x.nzind,y[x.nzind])
    @test_approx_eq(h_x,nx)
    y[x.nzind] = zero(elty)
    @test_approx_eq(h_y,y)
end

function test_gthrz(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseVector(x)
    d_y = CudaArray(y)
    d_z,d_w = CUSPARSE.gthrz(d_x,d_y,'O')
    h_w = to_host(d_w)
    h_z = to_host(d_z)
    nx = SparseVector(m,x.nzind,y[x.nzind])
    @test_approx_eq(h_z,nx)
    y[x.nzind] = zero(elty)
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
