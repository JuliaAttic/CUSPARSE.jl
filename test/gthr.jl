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
    if VERSION >= v"0.5.0-dev+742"
        @test h_x ≈ SparseVector(m,x.nzind,y[x.nzind])
    else
        @test h_x ≈ sparsevec(x.rowval,y[x.rowval],m)
    end
end

function test_gthr(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseVector(x)
    d_y = CudaArray(y)
    d_z = CUSPARSE.gthr(d_x,d_y,'O')
    h_z = to_host(d_z)
    if VERSION >= v"0.5.0-dev+742"
        @test h_z ≈ SparseVector(m,x.nzind,y[x.nzind])
    else
        @test h_z ≈ sparsevec(x.rowval,y[x.rowval],m)
    end
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
    if VERSION >= v"0.5.0-dev+742"
        @test h_x ≈ SparseVector(m,x.nzind,y[x.nzind])
        y[x.nzind] = zero(elty)
        @test h_y ≈ y
    else
        @test h_x ≈ sparsevec(x.rowval,y[x.rowval],m)
        y[x.rowval] = zero(elty)
        @test h_y ≈ y
    end
end

function test_gthrz(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseVector(x)
    d_y = CudaArray(y)
    d_z,d_w = CUSPARSE.gthrz(d_x,d_y,'O')
    h_w = to_host(d_w)
    h_z = to_host(d_z)
    if VERSION >= v"0.5.0-dev+742"
        @test h_z ≈ SparseVector(m,x.nzind,y[x.nzind])
        y[x.nzind] = zero(elty)
        @test h_w ≈ y
    else
        @test h_z ≈ sparsevec(x.rowval,y[x.rowval],m)
        y[x.rowval] = zero(elty)
        @test h_w ≈ y
    end
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
