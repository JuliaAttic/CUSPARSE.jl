using CUSPARSE
using CUDArt
using Base.Test

m = 20
n = 35
k = 13
d = 0.2

##############
# test_axpyi #
##############

function test_axpyi!(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrix(x)
    d_y = CudaArray(y)
    alpha = rand(elty)
    d_y = CUSPARSE.axpyi!(alpha,d_x,d_y,'O')
    #compare
    h_y = to_host(d_y)
    y[x.rowval] += alpha * x.nzval
    @test_approx_eq(h_y,y)
end
test_axpyi!(Float32)
test_axpyi!(Float64)
test_axpyi!(Complex64)
test_axpyi!(Complex128)

function test_axpyi(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrix(x)
    d_y = CudaArray(y)
    alpha = rand(elty)
    d_z = CUSPARSE.axpyi(alpha,d_x,d_y,'O')
    #compare
    h_z = to_host(d_z)
    z = copy(y)
    z[x.rowval] += alpha * x.nzval
    @test_approx_eq(h_z, z)
end
test_axpyi(Float32)
test_axpyi(Float64)
test_axpyi(Complex64)
test_axpyi(Complex128)

#############
# test_doti #
#############

function test_doti(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrix(x)
    d_y = CudaArray(y)
    ddot = CUSPARSE.doti(d_x,d_y,'O')
    #compare
    dot = zero(elty)
    for i in 1:length(x.nzval)
        dot += x.nzval[i] * y[x.rowval[i]]
    end
    @test_approx_eq(ddot, dot)
end
test_doti(Float32)
test_doti(Float64)
test_doti(Complex64)
test_doti(Complex128)

##############
# test_dotci #
##############

function test_dotci(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrix(x)
    d_y = CudaArray(y)
    ddot = CUSPARSE.dotci(d_x,d_y,'O')
    #compare
    dot = zero(elty)
    for i in 1:length(x.nzval)
        dot += conj(x.nzval[i]) * y[x.rowval[i]]
    end
    @test_approx_eq(ddot, dot)
end
test_dotci(Complex64)
test_dotci(Complex128)

#############
# test_gthr #
#############

function test_gthr!(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrix(x)
    d_y = CudaArray(y)
    d_y = CUSPARSE.gthr!(d_x,d_y,'O')
    h_x = to_host(d_x)
    x.nzval = y[x.rowval]
    @test_approx_eq(h_x,x)
end
test_gthr!(Float32)
test_gthr!(Float64)
test_gthr!(Complex64)
test_gthr!(Complex128)

function test_gthr(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrix(x)
    d_y = CudaArray(y)
    d_z = CUSPARSE.gthr(d_x,d_y,'O')
    h_z = to_host(d_z)
    x.nzval = y[x.rowval]
    @test_approx_eq(h_z,x)
end
test_gthr(Float32)
test_gthr(Float64)
test_gthr(Complex64)
test_gthr(Complex128)

##############
# test_gthrz #
##############

function test_gthrz!(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrix(x)
    d_y = CudaArray(y)
    d_x,d_y = CUSPARSE.gthrz!(d_x,d_y,'O')
    h_x = to_host(d_x)
    h_y = to_host(d_y)
    x.nzval = y[x.rowval]
    @test_approx_eq(h_x,x)
    y[x.rowval] = zero(elty)
    @test_approx_eq(h_y,y)
end
test_gthrz!(Float32)
test_gthrz!(Float64)
test_gthrz!(Complex64)
test_gthrz!(Complex128)

function test_gthrz(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrix(x)
    d_y = CudaArray(y)
    d_z,d_w = CUSPARSE.gthrz(d_x,d_y,'O')
    h_w = to_host(d_w)
    h_z = to_host(d_z)
    x.nzval = y[x.rowval]
    @test_approx_eq(h_z,x)
    y[x.rowval] = zero(elty)
    @test_approx_eq(h_w,y)
end
test_gthrz(Float32)
test_gthrz(Float64)
test_gthrz(Complex64)
test_gthrz(Complex128)

#############
# test_roti #
#############

function test_roti!(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrix(x)
    d_y = CudaArray(y)
    angle = rand(elty)
    d_x,d_y = CUSPARSE.roti!(d_x,d_y,cos(angle),sin(angle),'O')
    h_x = to_host(d_x)
    h_y = to_host(d_y)
    z = copy(x)
    w = copy(y)
    y[x.rowval] = cos(angle)*w[z.rowval] - sin(angle)*z.nzval
    x.nzval     = cos(angle)*z.nzval + sin(angle)*w[z.rowval]
    @test_approx_eq(h_y,y)
    @test_approx_eq(h_x,x)
end
test_roti!(Float32)
test_roti!(Float64)

function test_roti(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrix(x)
    d_y = CudaArray(y)
    angle = rand(elty)
    d_z,d_w = CUSPARSE.roti(d_x,d_y,cos(angle),sin(angle),'O')
    h_w = to_host(d_w)
    h_z = to_host(d_z)
    z = copy(x)
    w = copy(y)
    w[z.rowval] = cos(angle)*y[x.rowval] - sin(angle)*x.nzval
    z.nzval = cos(angle)*x.nzval + sin(angle)*y[x.rowval]
    @test_approx_eq(h_w,w)
    @test_approx_eq(h_z,z)
end
test_roti(Float32)
test_roti(Float64)

#############
# test_sctr #
#############

function test_sctr!(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = zeros(elty,m)
    d_x = CudaSparseMatrix(x)
    d_y = CudaArray(y)
    d_y = CUSPARSE.sctr!(d_x,d_y,'O')
    h_y = to_host(d_y)
    y[x.rowval] += x.nzval
    @test_approx_eq(h_y,y)
end
test_sctr!(Float32)
test_sctr!(Float64)
test_sctr!(Complex64)
test_sctr!(Complex128)

function test_sctr(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    d_x = CudaSparseMatrix(x)
    d_y = CUSPARSE.sctr(d_x,'O')
    h_y = to_host(d_y)
    y = zeros(elty,m)
    y[x.rowval] += x.nzval
    @test_approx_eq(h_y,y)
end
test_sctr(Float32)
test_sctr(Float64)
test_sctr(Complex64)
test_sctr(Complex128)
