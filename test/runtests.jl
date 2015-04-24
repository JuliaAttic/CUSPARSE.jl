using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
d = 0.2

# conversion
function test_make_csc(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSC(x)
    h_x = to_host(d_x)
    @test h_x == x
end
test_make_csc(Float32)
test_make_csc(Float64)
test_make_csc(Complex64)
test_make_csc(Complex128)

function test_make_csr(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSR(x)
    h_x = to_host(d_x)
    @test h_x.rowval == x.rowval
    @test_approx_eq(h_x.nzval,x.nzval)
end
test_make_csr(Float32)
test_make_csr(Float64)
test_make_csr(Complex64)
test_make_csr(Complex128)

function test_convert_r2c(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSR(x)
    d_x = CUSPARSE.switch(d_x)
    h_x = to_host(d_x)
    @test h_x.rowval == x.rowval
    @test_approx_eq(h_x.nzval,x.nzval)
end
test_convert_r2c(Float32)
test_convert_r2c(Float64)
test_convert_r2c(Complex64)
test_convert_r2c(Complex128)

function test_convert_c2r(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSC(x)
    d_x = CUSPARSE.switch(d_x)
    h_x = to_host(d_x)
    @test h_x.rowval == x.rowval
    @test_approx_eq(h_x.nzval,x.nzval)
end
test_convert_c2r(Float32)
test_convert_c2r(Float64)
test_convert_c2r(Complex64)
test_convert_c2r(Complex128)

function test_convert_r2d(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSR(x)
    d_x = CUSPARSE.full(d_x)
    h_x = to_host(d_x)
    @test_approx_eq(h_x,full(x))
end
test_convert_r2d(Float32)
test_convert_r2d(Float64)
test_convert_r2d(Complex64)
test_convert_r2d(Complex128)

function test_convert_c2d(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSC(x)
    d_x = CUSPARSE.full(d_x)
    h_x = to_host(d_x)
    @test_approx_eq(h_x,full(x))
end
test_convert_c2d(Float32)
test_convert_c2d(Float64)
test_convert_c2d(Complex64)
test_convert_c2d(Complex128)

function test_convert_d2c(elty)
    x = rand(elty,m,n)
    d_x = CudaArray(x)
    d_x = CUSPARSE.sparse(d_x,'C')
    h_x = to_host(d_x)
    @test_approx_eq(h_x,sparse(x))
end
test_convert_d2c(Float32)
test_convert_d2c(Float64)
test_convert_d2c(Complex64)
test_convert_d2c(Complex128)

function test_convert_d2r(elty)
    x = rand(elty,m,n)
    d_x = CudaArray(x)
    d_x = CUSPARSE.sparse(d_x)
    h_x = to_host(d_x)
    @test_approx_eq(h_x,sparse(x))
end
test_convert_d2r(Float32)
test_convert_d2r(Float64)
test_convert_d2r(Complex64)
test_convert_d2r(Complex128)

# level 1 functions

##############
# test_axpyi #
##############

function test_axpyi!(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrixCSC(x)
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
#test_axpyi!(Complex128)

function test_axpyi(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrixCSC(x)
    d_y = CudaArray(y)
    alpha = rand(elty)
    d_z = CUSPARSE.axpyi(alpha,d_x,d_y,'O')
    #compare
    h_z = to_host(d_z)
    z = copy(y)
    z[x.rowval] += alpha * x.nzval
    @test_approx_eq(h_z, z)
    d_z = CUSPARSE.axpyi(d_x,d_y,'O')
    #compare
    h_z = to_host(d_z)
    z = copy(y)
    z[x.rowval] += x.nzval
    @test_approx_eq(h_z, z)
end
test_axpyi(Float32)
test_axpyi(Float64)
test_axpyi(Complex64)
#test_axpyi(Complex128)

#############
# test_doti #
#############

function test_doti(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    y = rand(elty,m)
    d_x = CudaSparseMatrixCSC(x)
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
    d_x = CudaSparseMatrixCSC(x)
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
    d_x = CudaSparseMatrixCSC(x)
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
    d_x = CudaSparseMatrixCSC(x)
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
test_gthrz!(Float32)
test_gthrz!(Float64)
test_gthrz!(Complex64)
test_gthrz!(Complex128)

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
    d_x = CudaSparseMatrixCSC(x)
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
    d_x = CudaSparseMatrixCSC(x)
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
    d_x = CudaSparseMatrixCSC(x)
    d_y = CudaArray(y)
    d_y = CUSPARSE.sctr!(d_x,d_y,'O')
    h_y = to_host(d_y)
    y[x.rowval]  += x.nzval
    @test_approx_eq(h_y,y)
end
test_sctr!(Float32)
test_sctr!(Float64)
test_sctr!(Complex64)
#test_sctr!(Complex128)

function test_sctr(elty)
    x = sparsevec(rand(1:m,k), rand(elty,k), m)
    d_x = CudaSparseMatrixCSC(x)
    d_y = CUSPARSE.sctr(d_x,'O')
    h_y = to_host(d_y)
    y = zeros(elty,m)
    y[x.rowval] += x.nzval
    @test_approx_eq(h_y,y)
end
test_sctr(Float32)
test_sctr(Float64)
test_sctr(Complex64)
#test_sctr(Complex128)

## level 2

##############
# test_csrmv #
##############

function test_csrmv!(elty)
    A = sparse(rand(elty,m,n))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    d_A = CudaSparseMatrixCSR(A)
    d_y = CUSPARSE.csrmv!('N',alpha,d_A,d_x,beta,d_y,'O')
    h_y = to_host(d_y)
    y = alpha * A * x + beta * y
    @test_approx_eq(y,h_y)
    @test_throws(DimensionMismatch, CUSPARSE.csrmv!('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.csrmv!('N',alpha,d_A,d_y,beta,d_x,'O'))
end
test_csrmv!(Float32)
test_csrmv!(Float64)
test_csrmv!(Complex64)
test_csrmv!(Complex128)

function test_csrmv(elty)
    A = sparse(rand(elty,m,n))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    d_A = CudaSparseMatrixCSR(A)
    d_z = CUSPARSE.csrmv('N',alpha,d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.csrmv('N',d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.csrmv('N',d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.csrmv('N',alpha,d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.csrmv('N',alpha,d_A,d_x,'O')
    h_z = to_host(d_z)
    z = alpha * A * x
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.csrmv('N',d_A,d_x,'O')
    h_z = to_host(d_z)
    z = A * x
    @test_approx_eq(z,h_z)
    @test_throws(DimensionMismatch, CUSPARSE.csrmv('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.csrmv('N',alpha,d_A,d_y,beta,d_x,'O'))
end
test_csrmv(Float32)
test_csrmv(Float64)
test_csrmv(Complex64)
test_csrmv(Complex128)

## level 3

##############
# test_csrmm #
##############

function test_csrmm!(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_C = CUSPARSE.csrmm!('N',alpha,d_A,d_B,beta,d_C,'O')
    h_C = to_host(d_C)
    C = alpha * A * B + beta * C
    @test_approx_eq(C,h_C)
    @test_throws(DimensionMismatch, CUSPARSE.csrmm!('T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.csrmm!('N',alpha,d_A,d_B,beta,d_B,'O'))
end
test_csrmm!(Float32)
test_csrmm!(Float64)
test_csrmm!(Complex64)
test_csrmm!(Complex128)

function test_csrmm(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_D = CUSPARSE.csrmm('N',alpha,d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = alpha * A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm('N',d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm('N',d_A,d_B,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm('N',alpha,d_A,d_B,'O')
    h_D = to_host(d_D)
    D = alpha * A * B
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm('N',d_A,d_B,'O')
    h_D = to_host(d_D)
    D = A * B
    @test_approx_eq(D,h_D)
    @test_throws(DimensionMismatch, CUSPARSE.csrmm('T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.csrmm('N',alpha,d_A,d_B,beta,d_B,'O'))
end
test_csrmm(Float32)
test_csrmm(Float64)
test_csrmm(Complex64)
test_csrmm(Complex128)

###############
# test_csrmm2 #
###############

function test_csrmm2!(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_C = CUSPARSE.csrmm2!('N','N',alpha,d_A,d_B,beta,d_C,'O')
    h_C = to_host(d_C)
    C = alpha * A * B + beta * C
    @test_approx_eq(C,h_C)
    @test_throws(DimensionMismatch, CUSPARSE.csrmm2!('N','T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.csrmm2!('T','N',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.csrmm2!('T','T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.csrmm2!('N','N',alpha,d_A,d_B,beta,d_B,'O'))
end
test_csrmm2!(Float32)
test_csrmm2!(Float64)
test_csrmm2!(Complex64)
test_csrmm2!(Complex128)

function test_csrmm2(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_D = CUSPARSE.csrmm2('N','N',alpha,d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = alpha * A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm2('N','N',d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm2('N','N',d_A,d_B,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm2('N','N',alpha,d_A,d_B,'O')
    h_D = to_host(d_D)
    D = alpha * A * B
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.csrmm2('N','N',d_A,d_B,'O')
    h_D = to_host(d_D)
    D = A * B
    @test_approx_eq(D,h_D)
end
test_csrmm2(Float32)
test_csrmm2(Float64)
test_csrmm2(Complex64)
test_csrmm2(Complex128)

## extensions

#############
# test_geam #
#############

function test_geam(elty)
    A = sparse(rand(elty,m,n))
    B = sparse(rand(elty,m,n))
    alpha = rand(elty)
    beta = rand(elty)
    C = alpha * A + beta * B
    d_A = CudaSparseMatrixCSR(A)
    d_B = CudaSparseMatrixCSR(B)
    d_C = CUSPARSE.geam(alpha,d_A,beta,d_B,'O','O','O')
    h_C = to_host(d_C)
    @test_approx_eq(C,h_C)
    d_C = CUSPARSE.geam(d_A,beta,d_B,'O','O','O')
    h_C = to_host(d_C)
    C = A + beta * B
    @test_approx_eq(C,h_C)
    d_C = CUSPARSE.geam(alpha,d_A,d_B,'O','O','O')
    h_C = to_host(d_C)
    C = alpha * A + B
    @test_approx_eq(C,h_C)
    d_C = CUSPARSE.geam(d_A,d_B,'O','O','O')
    h_C = to_host(d_C)
    C = A + B
    @test_approx_eq(C,h_C)
    B = sparse(rand(elty,k,n))
    d_B = CudaSparseMatrixCSR(B)
    @test_throws(DimensionMismatch,CUSPARSE.geam(d_B,d_A,'O','O','O'))
end
test_geam(Float32)
test_geam(Float64)
test_geam(Complex64)
test_geam(Complex128)

#############
# test_gemm #
#############

function test_gemm(elty)
    A = sparse(rand(elty,m,k))
    B = sparse(rand(elty,k,n))
    C = A * B
    d_A = CudaSparseMatrixCSR(A)
    d_B = CudaSparseMatrixCSR(B)
    d_C = CUSPARSE.gemm('N','N',d_A,d_B,'O','O','O')
    h_C = to_host(d_C)
    @test_approx_eq(C,h_C)
    @test_throws(DimensionMismatch,CUSPARSE.gemm('N','T',d_A,d_B,'O','O','O'))
    @test_throws(DimensionMismatch,CUSPARSE.gemm('T','T',d_A,d_B,'O','O','O'))
    @test_throws(DimensionMismatch,CUSPARSE.gemm('T','N',d_A,d_B,'O','O','O'))
    @test_throws(DimensionMismatch,CUSPARSE.gemm('N','N',d_B,d_A,'O','O','O'))
end
test_gemm(Float32)
test_gemm(Float64)
test_gemm(Complex64)
test_gemm(Complex128)
