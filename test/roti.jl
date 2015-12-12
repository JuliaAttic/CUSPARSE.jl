using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

#############
# test_roti #
#############

function test_roti!(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseVector(x)
    d_y = CudaArray(y)
    angle = rand(elty)
    d_x,d_y = CUSPARSE.roti!(d_x,d_y,cos(angle),sin(angle),'O')
    h_x = to_host(d_x)
    h_y = to_host(d_y)
    z = copy(x)
    w = copy(y)
    y[x.nzind] = cos(angle)*w[z.nzind] - sin(angle)*z.nzval
    nx = SparseVector(m,x.nzind,cos(angle)*z.nzval + sin(angle)*w[z.nzind])
    @test_approx_eq(h_y,y)
    @test_approx_eq(h_x,nx)
end

function test_roti(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseVector(x)
    d_y = CudaArray(y)
    angle = rand(elty)
    d_z,d_w = CUSPARSE.roti(d_x,d_y,cos(angle),sin(angle),'O')
    h_w = to_host(d_w)
    h_z = to_host(d_z)
    z = copy(x)
    w = copy(y)
    w[z.nzind] = cos(angle)*y[x.nzind] - sin(angle)*x.nzval
    nz = SparseVector(m,z.nzind, cos(angle)*x.nzval + sin(angle)*y[x.nzind])
    @test_approx_eq(h_w,w)
    @test_approx_eq(h_z,nz)
end

types = [Float32,Float64]
for elty in types
    tic()
    test_roti!(elty)
    println("roti! took ", toq(), " for ", elty)
    tic()
    test_roti(elty)
    println("roti took ", toq(), " for ", elty)
end
