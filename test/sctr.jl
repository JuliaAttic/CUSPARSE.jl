using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

#############
# test_sctr #
#############

function test_sctr!(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = zeros(elty,m)
    d_x = CudaSparseVector(x)
    d_y = CudaArray(y)
    d_y = CUSPARSE.sctr!(d_x,d_y,'O')
    h_y = to_host(d_y)
    if VERSION >= v"0.5.0-dev+742"
        y[x.nzind]  += x.nzval
    else
        y[x.rowval] += x.nzval
    end
    @test h_y ≈ y
end

function test_sctr(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    d_x = CudaSparseVector(x)
    d_y = CUSPARSE.sctr(d_x,'O')
    h_y = to_host(d_y)
    y = zeros(elty,m)
    if VERSION >= v"0.5.0-dev+742"
        y[x.nzind]  += x.nzval
    else
        y[x.rowval] += x.nzval
    end
    @test h_y ≈ y
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_sctr!(elty)
    println("sctr! took ", toq(), " for ", elty)
    tic()
    test_sctr(elty)
    println("sctr took ", toq(), " for ", elty)
end
