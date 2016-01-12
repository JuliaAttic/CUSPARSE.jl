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
    if VERSION >= v"0.5.0-dev+742"
        y[x.nzind] = cos(angle)*w[z.nzind] - sin(angle)*z.nzval
        @test h_x ≈ SparseVector(m,x.nzind,cos(angle)*z.nzval + sin(angle)*w[z.nzind])
    else
        y[x.rowval] = cos(angle)*w[z.rowval] - sin(angle)*z.nzval
        @test h_x ≈ sparsevec(x.rowval,cos(angle)*z.nzval + sin(angle)*w[z.rowval],m)
    end
    @test h_y ≈ y
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
    if VERSION >= v"0.5.0-dev+742"
        w[z.nzind] = cos(angle)*y[x.nzind] - sin(angle)*x.nzval
        @test h_z ≈ SparseVector(m,z.nzind, cos(angle)*x.nzval + sin(angle)*y[x.nzind])
    else
        w[z.rowval] = cos(angle)*y[x.rowval] - sin(angle)*x.nzval
        @test h_z ≈ sparsevec(z.rowval, cos(angle)*x.nzval + sin(angle)*y[x.rowval],m)
    end
    @test h_w ≈ w
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
