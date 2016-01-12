using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

##############
# test_axpyi #
##############
function test_axpyi!(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseVector(x)
    d_y = CudaArray(y)
    alpha = rand(elty)
    d_y = CUSPARSE.axpyi!(alpha,d_x,d_y,'O')
    #compare
    h_y = to_host(d_y)
    if VERSION >= v"0.5.0-dev+742"
        y[x.nzind] += alpha * x.nzval
    else
        y[x.rowval] += alpha * x.nzval
    end
    @test h_y ≈ y
end

function test_axpyi(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseVector(x)
    d_y = CudaArray(y)
    alpha = rand(elty)
    d_z = CUSPARSE.axpyi(alpha,d_x,d_y,'O')
    #compare
    h_z = to_host(d_z)
    z = copy(y)
    if VERSION >= v"0.5.0-dev+742"
        z[x.nzind] += alpha * x.nzval
    else
        z[x.rowval] += alpha * x.nzval
    end
    @test h_z ≈ z
    d_z = CUSPARSE.axpyi(d_x,d_y,'O')
    #compare
    h_z = to_host(d_z)
    z = copy(y)
    if VERSION >= v"0.5.0-dev+742"
        z[x.nzind] += x.nzval
    else
        z[x.rowval] += x.nzval
    end
    @test h_z ≈ z
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_axpyi!(elty)
    println("axpyi! took ", toq(), " for ", elty)
    tic()
    test_axpyi(elty)
    println("axpyi took ", toq(), " for ", elty)
end
