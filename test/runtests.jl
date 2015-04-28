using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

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
    d_x = CUSPARSE.switch2csc(d_x)
    h_x = to_host(d_x)
    @test h_x.rowval == x.rowval
    @test_approx_eq(h_x.nzval,x.nzval)
end
test_convert_r2c(Float32)
test_convert_r2c(Float64)
test_convert_r2c(Complex64)
test_convert_r2c(Complex128)

function test_convert_r2b(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSR(x)
    d_x = CUSPARSE.switch2bsr(d_x,convert(Cint,blockdim))
    d_x = CUSPARSE.switch2csr(d_x)
    h_x = to_host(d_x)
    @test_approx_eq(h_x,x)
end
test_convert_r2b(Float32)
test_convert_r2b(Float64)
test_convert_r2b(Complex64)
test_convert_r2b(Complex128)

function test_convert_c2b(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSC(x)
    d_x = CUSPARSE.switch2bsr(d_x,convert(Cint,blockdim))
    d_x = CUSPARSE.switch2csc(d_x)
    h_x = to_host(d_x)
    @test_approx_eq(h_x,x)
end
test_convert_c2b(Float32)
test_convert_c2b(Float64)
test_convert_c2b(Complex64)
test_convert_c2b(Complex128)

function test_convert_c2h(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSC(x)
    d_x = CUSPARSE.switch2hyb(d_x)
    d_y = CUSPARSE.switch2csc(d_x)
    CUSPARSE.cusparseDestroyHybMat(d_x.Mat)
    h_x = to_host(d_y)
    @test h_x.rowval == x.rowval
    @test_approx_eq(h_x.nzval,x.nzval)
end
test_convert_c2h(Float32)
test_convert_c2h(Float64)
test_convert_c2h(Complex64)
test_convert_c2h(Complex128)

function test_convert_r2h(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSR(x)
    d_x = CUSPARSE.switch2hyb(d_x)
    d_y = CUSPARSE.switch2csr(d_x)
    CUSPARSE.cusparseDestroyHybMat(d_x.Mat)
    h_x = to_host(d_y)
    @test h_x.rowval == x.rowval
    @test_approx_eq(h_x.nzval,x.nzval)
end
test_convert_r2h(Float32)
test_convert_r2h(Float64)
test_convert_r2h(Complex64)
test_convert_r2h(Complex128)

function test_convert_d2h(elty)
    x = rand(elty,m,n)
    d_x = CudaArray(x)
    d_x = CUSPARSE.sparse(d_x,'H')
    d_y = CUSPARSE.full(d_x)
    CUSPARSE.cusparseDestroyHybMat(d_x.Mat)
    h_x = to_host(d_y)
    @test_approx_eq(h_x,x)
end
test_convert_d2h(Float32)
test_convert_d2h(Float64)
test_convert_d2h(Complex64)
test_convert_d2h(Complex128)

function test_convert_d2b(elty)
    x = rand(elty,m,n)
    d_x = CudaArray(x)
    d_x = CUSPARSE.sparse(d_x,'B')
    d_y = CUSPARSE.full(d_x)
    h_x = to_host(d_y)
    @test_approx_eq(h_x,x)
end
test_convert_d2b(Float32)
test_convert_d2b(Float64)
test_convert_d2b(Complex64)
test_convert_d2b(Complex128)

function test_convert_c2r(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSC(x)
    d_x = CUSPARSE.switch2csr(d_x)
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
# test_bsrmv #
##############

function test_bsrmv!(elty)
    A = sparse(rand(elty,m,n))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
    d_y = CUSPARSE.bsrmv!('N',alpha,d_A,d_x,beta,d_y,'O')
    h_y = to_host(d_y)
    y = alpha * A * x + beta * y
    @test_approx_eq(y,h_y)
    @test_throws(DimensionMismatch, CUSPARSE.bsrmv!('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.bsrmv!('N',alpha,d_A,d_y,beta,d_x,'O'))
end
test_bsrmv!(Float32)
test_bsrmv!(Float64)
test_bsrmv!(Complex64)
test_bsrmv!(Complex128)

function test_bsrmv(elty)
    A = sparse(rand(elty,m,n))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
    d_z = CUSPARSE.bsrmv('N',alpha,d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.bsrmv('N',d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.bsrmv('N',d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.bsrmv('N',alpha,d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.bsrmv('N',alpha,d_A,d_x,'O')
    h_z = to_host(d_z)
    z = alpha * A * x
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.bsrmv('N',d_A,d_x,'O')
    h_z = to_host(d_z)
    z = A * x
    @test_approx_eq(z,h_z)
    @test_throws(DimensionMismatch, CUSPARSE.bsrmv('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.bsrmv('N',alpha,d_A,d_y,beta,d_x,'O'))
end
test_bsrmv(Float32)
test_bsrmv(Float64)
test_bsrmv(Complex64)
test_bsrmv(Complex128)

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

###############
# test_bsrsv2 #
###############

function test_bsrsv2!(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    d_X = CUSPARSE.bsrsv2!('N',alpha,d_A,d_X,'O')
    h_Y = to_host(d_X)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    @test_throws(DimensionMismatch, CUSPARSE.bsrsv2!('N',alpha,d_A,d_X,'O'))
end
test_bsrsv2!(Float32)
test_bsrsv2!(Float64)
test_bsrsv2!(Complex64)
test_bsrsv2!(Complex128)

function test_bsrsv2(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    d_Y = CUSPARSE.bsrsv2('N',alpha,d_A,d_X,'O')
    h_Y = to_host(d_Y)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    @test_throws(DimensionMismatch, CUSPARSE.bsrsv2('N',alpha,d_A,d_X,'O'))
end
test_bsrsv2(Float32)
test_bsrsv2(Float64)
test_bsrsv2(Complex64)
test_bsrsv2(Complex128)

##############
# test_csrsv #
##############

function test_csrsv(elty)
    A = rand(elty,m,m)
    A = triu(A)
    x = rand(elty,m)
    alpha = rand(elty)
    d_x = CudaArray(x)
    d_A = CudaSparseMatrixCSR(sparse(A))
    info = CUSPARSE.csrsv_analysis('N','T','U',d_A,'O')
    d_y = CUSPARSE.csrsv_solve('N','U',alpha,d_A,d_x,info,'O')
    h_y = to_host(d_y)
    y = A\(alpha * x)
    @test_approx_eq(y,h_y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    @test_throws(DimensionMismatch, CUSPARSE.csrsv_analysis('T','T','U',d_A,'O'))
    CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
end
test_csrsv(Float32)
test_csrsv(Float64)
test_csrsv(Complex64)
test_csrsv(Complex128)

###############
# test_csrsv2 #
###############

function test_csrsv2!(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_X = CUSPARSE.csrsv2!('N',alpha,d_A,d_X,'O')
    h_Y = to_host(d_X)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(sparse(A))
    @test_throws(DimensionMismatch, CUSPARSE.csrsv2!('N',alpha,d_A,d_X,'O'))
end
test_csrsv2!(Float32)
test_csrsv2!(Float64)
test_csrsv2!(Complex64)
test_csrsv2!(Complex128)

function test_csrsv2(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_Y = CUSPARSE.csrsv2('N',alpha,d_A,d_X,'O')
    h_Y = to_host(d_Y)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    @test_throws(DimensionMismatch, CUSPARSE.csrsv2('N',alpha,d_A,d_X,'O'))
end
test_csrsv2(Float32)
test_csrsv2(Float64)
test_csrsv2(Complex64)
test_csrsv2(Complex128)

##############
# test_hybmv #
##############

function test_hybmv!(elty)
    A = sparse(rand(elty,m,n))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2hyb(d_A)
    d_y = CUSPARSE.hybmv!('N',alpha,d_A,d_x,beta,d_y,'O')
    h_y = to_host(d_y)
    y = alpha * A * x + beta * y
    @test_approx_eq(y,h_y)
    @test_throws(DimensionMismatch, CUSPARSE.hybmv!('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.hybmv!('N',alpha,d_A,d_y,beta,d_x,'O'))
end
test_hybmv!(Float32)
test_hybmv!(Float64)
test_hybmv!(Complex64)
test_hybmv!(Complex128)

function test_hybmv(elty)
    A = sparse(rand(elty,m,n))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2hyb(d_A)
    d_z = CUSPARSE.hybmv('N',alpha,d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.hybmv('N',d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.hybmv('N',d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.hybmv('N',alpha,d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.hybmv('N',alpha,d_A,d_x,'O')
    h_z = to_host(d_z)
    z = alpha * A * x
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.hybmv('N',d_A,d_x,'O')
    h_z = to_host(d_z)
    z = A * x
    @test_approx_eq(z,h_z)
    @test_throws(DimensionMismatch, CUSPARSE.hybmv('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.hybmv('N',alpha,d_A,d_y,beta,d_x,'O'))
end
test_hybmv(Float32)
test_hybmv(Float64)
test_hybmv(Complex64)
test_hybmv(Complex128)

##############
# test_hybsv #
##############

function test_hybsv(elty)
    A = rand(elty,m,m)
    A = triu(A)
    x = rand(elty,m)
    alpha = rand(elty)
    beta = rand(elty)
    d_x = CudaArray(x)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_A = CUSPARSE.switch2hyb(d_A)
    info = CUSPARSE.hybsv_analysis('N','U',d_A,'O')
    d_y = CUSPARSE.hybsv_solve('N','U',alpha,d_A,d_x,info,'O')
    h_y = to_host(d_y)
    y = A\(alpha * x)
    @test_approx_eq(y,h_y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2hyb(d_A)
    @test_throws(DimensionMismatch, CUSPARSE.hybsv_analysis('T','U',d_A,'O'))
    CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
end
test_hybsv(Float32)
test_hybsv(Float64)
test_hybsv(Complex64)
test_hybsv(Complex128)

## level 3

###############
# test_bsrmm #
###############

function test_bsrmm!(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
    d_C = CUSPARSE.bsrmm!('N','N',alpha,d_A,d_B,beta,d_C,'O')
    h_C = to_host(d_C)
    C = alpha * A * B + beta * C
    @test_approx_eq(C,h_C)
    @test_throws(DimensionMismatch, CUSPARSE.bsrmm!('N','T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.bsrmm!('T','N',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.bsrmm!('T','T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.bsrmm!('N','N',alpha,d_A,d_B,beta,d_B,'O'))
end
test_bsrmm!(Float32)
test_bsrmm!(Float64)
test_bsrmm!(Complex64)
test_bsrmm!(Complex128)

function test_bsrmm(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
    d_D = CUSPARSE.bsrmm('N','N',alpha,d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = alpha * A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.bsrmm('N','N',d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.bsrmm('N','N',d_A,d_B,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.bsrmm('N','N',alpha,d_A,d_B,'O')
    h_D = to_host(d_D)
    D = alpha * A * B
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.bsrmm('N','N',d_A,d_B,'O')
    h_D = to_host(d_D)
    D = A * B
    @test_approx_eq(D,h_D)
end
test_bsrmm(Float32)
test_bsrmm(Float64)
test_bsrmm(Complex64)
test_bsrmm(Complex128)

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

##############
# test_csrsm #
##############

function test_csrsm(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m,n)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSR(sparse(A))
    info = CUSPARSE.csrsm_analysis('N','U',d_A,'O')
    d_Y = CUSPARSE.csrsm_solve('N','U',alpha,d_A,d_X,info,'O')
    h_Y = to_host(d_Y)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    @test_throws(DimensionMismatch, CUSPARSE.csrsm_analysis('T','U',d_A,'O'))
    CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
end
test_csrsm(Float32)
test_csrsm(Float64)
test_csrsm(Complex64)
test_csrsm(Complex128)

###############
# test_bsrsm2 #
###############

function test_bsrsm2!(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m,n)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    d_X = CUSPARSE.bsrsm2!('N','N',alpha,d_A,d_X,'O')
    h_Y = to_host(d_X)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    @test_throws(DimensionMismatch, CUSPARSE.bsrsm2!('N','N',alpha,d_A,d_X,'O'))
end
test_bsrsm2!(Float32)
test_bsrsm2!(Float64)
test_bsrsm2!(Complex64)
test_bsrsm2!(Complex128)

function test_bsrsm2(elty)
    A = rand(elty,m,m)
    A = triu(A)
    X = rand(elty,m,n)
    alpha = rand(elty)
    d_X = CudaArray(X)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    d_Y = CUSPARSE.bsrsm2('N','N',alpha,d_A,d_X,'O')
    h_Y = to_host(d_Y)
    Y = A\(alpha * X)
    @test_approx_eq(Y,h_Y)
    A = sparse(rand(elty,m,n))
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    @test_throws(DimensionMismatch, CUSPARSE.bsrsm2('N','N',alpha,d_A,d_X,'O'))
end
test_bsrsm2(Float32)
test_bsrsm2(Float64)
test_bsrsm2(Complex64)
test_bsrsm2(Complex128)

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

## preconditioners

###############
# test_csric0 #
###############

function test_csric0_real!(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(triu(A)))
    info = CUSPARSE.csrsv_analysis('N','S','U',d_A,'O')
    d_A = CUSPARSE.csric0!('N','S',d_A,info,'O')
    h_A = to_host(d_A)
    Ac = sparse(full(cholfact(A)))
    h_A = transpose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
end
test_csric0_real!(Float32)
test_csric0_real!(Float64)

function test_csric0_real(elty)
    A = rand(elty,m,m)
    A = A + transpose(A) 
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(triu(A)))
    info = CUSPARSE.csrsv_analysis('N','S','U',d_A,'O')
    d_B = CUSPARSE.csric0('N','S',d_A,info,'O')
    h_A = to_host(d_B)
    Ac = sparse(full(cholfact(A)))
    h_A = transpose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
end
test_csric0_real(Float32)
test_csric0_real(Float64)

function test_csric0_complex!(elty)
    A = rand(elty,m,m)
    A = A + ctranspose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(triu(A)))
    info = CUSPARSE.csrsv_analysis('N','H','U',d_A,'O')
    d_A = CUSPARSE.csric0!('N','H',d_A,info,'O')
    h_A = to_host(d_A)
    Ac = sparse(full(cholfact(A)))
    h_A = ctranspose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
end
test_csric0_complex!(Complex64)
test_csric0_complex!(Complex128)

function test_csric0_complex(elty)
    A = rand(elty,m,m)
    A = A + ctranspose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(triu(A)))
    info = CUSPARSE.csrsv_analysis('N','H','U',d_A,'O')
    d_B = CUSPARSE.csric0('N','H',d_A,info,'O')
    h_A = to_host(d_B)
    Ac = sparse(full(cholfact(A)))
    h_A = ctranspose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
end
test_csric0_complex(Complex64)
test_csric0_complex(Complex128)

################
# test_csric02 #
################

function test_csric02!(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(tril(A)))
    d_A = CUSPARSE.csric02!(d_A,'O')
    h_A = to_host(d_A)
    Ac = sparse(full(cholfact(A)))
    h_A = transpose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
end
test_csric02!(Float32)
test_csric02!(Float64)
test_csric02!(Complex64)
test_csric02!(Complex128)

function test_csric02(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(tril(A)))
    d_B = CUSPARSE.csric02(d_A,'O')
    h_A = to_host(d_B)
    Ac = sparse(full(cholfact(A)))
    h_A = transpose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
end
test_csric02(Float32)
test_csric02(Float64)
test_csric02(Complex64)
test_csric02(Complex128)

################
# test_csrilu0 #
################

function test_csrilu0!(elty)
    A = sparse(rand(elty,m,m))
    d_A = CudaSparseMatrixCSR(A)
    info = CUSPARSE.csrsv_analysis('N','G','U',d_A,'O')
    d_A = CUSPARSE.csrilu0!('N',d_A,info,'O')
    h_A = to_host(d_A)
    Alu = lufact(full(A),pivot=false)
    Ac = sparse(Alu[:L]*Alu[:U])
    h_A = ctranspose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
end
test_csrilu0!(Float32)
test_csrilu0!(Float64)
test_csrilu0!(Complex64)
test_csrilu0!(Complex128)

function test_csrilu0(elty)
    A = rand(elty,m,m)
    d_A = CudaSparseMatrixCSR(sparse(A))
    info = CUSPARSE.csrsv_analysis('N','G','U',d_A,'O')
    d_B = CUSPARSE.csrilu0('N',d_A,info,'O')
    h_B = to_host(d_B)
    Alu = lufact(full(A),pivot=false)
    Ac = sparse(Alu[:L]*Alu[:U])
    h_A = ctranspose(h_B) * h_B
    @test_approx_eq(h_A.rowval,Ac.rowval)
end
test_csrilu0(Float32)
test_csrilu0(Float64)
test_csrilu0(Complex64)
test_csrilu0(Complex128)

################
# test_csrilu02 #
################

function test_csrilu02!(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_A = CUSPARSE.csrilu02!(d_A,'O')
    h_A = to_host(d_A)
    Alu = lufact(full(A),pivot=false)
    Ac = sparse(Alu[:L]*Alu[:U])
    h_A = ctranspose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
end
test_csrilu02!(Float32)
test_csrilu02!(Float64)
test_csrilu02!(Complex64)
test_csrilu02!(Complex128)

function test_csrilu02(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_B = CUSPARSE.csrilu02(d_A,'O')
    h_A = to_host(d_B)
    Alu = lufact(full(A),pivot=false)
    Ac = sparse(Alu[:L]*Alu[:U])
    h_A = ctranspose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
end
test_csrilu02(Float32)
test_csrilu02(Float64)
test_csrilu02(Complex64)
test_csrilu02(Complex128)

################
# test_bsric02 #
################

function test_bsric02!(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(tril(A)))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    d_A = CUSPARSE.bsric02!(d_A,'O')
    h_A = to_host(CUSPARSE.switch2csr(d_A))
    Ac = sparse(full(cholfact(A)))
    h_A = transpose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
end
test_bsric02!(Float32)
test_bsric02!(Float64)
test_bsric02!(Complex64)
test_bsric02!(Complex128)

function test_bsric02(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(tril(A)))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    d_B = CUSPARSE.bsric02(d_A,'O')
    h_A = to_host(CUSPARSE.switch2csr(d_B))
    Ac = sparse(full(cholfact(A)))
    h_A = transpose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
end
test_bsric02(Float32)
test_bsric02(Float64)
test_bsric02(Complex64)
test_bsric02(Complex128)

#############
# test_gtsv #
#############

function test_gtsv!(elty)
    dl = rand(elty,m-1)
    du = rand(elty,m-1)
    d = rand(elty,m)
    B = rand(elty,m,n)
    d_dl = CudaArray(vcat([0],dl))
    d_du = CudaArray(vcat(du,[0]))
    d_d = CudaArray(d)
    d_B = CudaArray(B)
    d_B = CUSPARSE.gtsv!(d_dl,d_d,d_du,d_B)
    C = diagm(d,0) + diagm(du,1) + diagm(dl,-1)
    h_B = to_host(d_B)
    @test_approx_eq(h_B, C\B)
end
test_gtsv!(Float32)
test_gtsv!(Float64)
test_gtsv!(Complex64)
test_gtsv!(Complex128)

function test_gtsv(elty)
    dl = rand(elty,m-1)
    du = rand(elty,m-1)
    d = rand(elty,m)
    B = rand(elty,m,n)
    d_dl = CudaArray(vcat([0],dl))
    d_du = CudaArray(vcat(du,[0]))
    d_d = CudaArray(d)
    d_B = CudaArray(B)
    d_C = CUSPARSE.gtsv(d_dl,d_d,d_du,d_B)
    C = diagm(d,0) + diagm(du,1) + diagm(dl,-1)
    h_C = to_host(d_C)
    @test_approx_eq(h_C, C\B)
end
test_gtsv(Float32)
test_gtsv(Float64)
test_gtsv(Complex64)
test_gtsv(Complex128)

#####################
# test_gtsv_nopivot #
#####################

function test_gtsv_nopivot!(elty)
    dl = rand(elty,m-1)
    du = rand(elty,m-1)
    d = rand(elty,m)
    B = rand(elty,m,n)
    d_dl = CudaArray(vcat([0],dl))
    d_du = CudaArray(vcat(du,[0]))
    d_d = CudaArray(d)
    d_B = CudaArray(B)
    d_B = CUSPARSE.gtsv_nopivot!(d_dl,d_d,d_du,d_B)
    C = diagm(d,0) + diagm(du,1) + diagm(dl,-1)
    h_B = to_host(d_B)
    @test_approx_eq(h_B, C\B)
end
test_gtsv_nopivot!(Float32)
test_gtsv_nopivot!(Float64)
test_gtsv_nopivot!(Complex64)
test_gtsv_nopivot!(Complex128)

function test_gtsv_nopivot(elty)
    dl = rand(elty,m-1)
    du = rand(elty,m-1)
    d = rand(elty,m)
    B = rand(elty,m,n)
    d_dl = CudaArray(vcat([0],dl))
    d_du = CudaArray(vcat(du,[0]))
    d_d = CudaArray(d)
    d_B = CudaArray(B)
    d_C = CUSPARSE.gtsv_nopivot(d_dl,d_d,d_du,d_B)
    C = diagm(d,0) + diagm(du,1) + diagm(dl,-1)
    h_C = to_host(d_C)
    @test_approx_eq(h_C, C\B)
end
test_gtsv_nopivot(Float32)
test_gtsv_nopivot(Float64)
test_gtsv_nopivot(Complex64)
test_gtsv_nopivot(Complex128)

#########################
# test_gtsvStridedBatch #
#########################

function test_gtsvStridedBatch!(elty)
    dla = rand(elty,m-1)
    dua = rand(elty,m-1)
    da = rand(elty,m)
    dlb = rand(elty,m-1)
    dub = rand(elty,m-1)
    db = rand(elty,m)
    xa = rand(elty,m)
    xb = rand(elty,m)
    d_dl = CudaArray(vcat([0],dla,[0],dlb))
    d_du = CudaArray(vcat(dua,[0],dub,[0]))
    d_d = CudaArray(vcat(da,db))
    d_x = CudaArray(vcat(xa,xb))
    d_x = CUSPARSE.gtsvStridedBatch!(d_dl,d_d,d_du,d_x,2,m)
    Ca = diagm(da,0) + diagm(dua,1) + diagm(dla,-1)
    Cb = diagm(db,0) + diagm(dub,1) + diagm(dlb,-1)
    h_x = to_host(d_x)
    @test_approx_eq(h_x[1:m], Ca\xa)
    @test_approx_eq(h_x[m+1:2*m], Cb\xb)
end
test_gtsvStridedBatch!(Float32)
test_gtsvStridedBatch!(Float64)
test_gtsvStridedBatch!(Complex64)
test_gtsvStridedBatch!(Complex128)

function test_gtsvStridedBatch(elty)
    dla = rand(elty,m-1)
    dua = rand(elty,m-1)
    da = rand(elty,m)
    dlb = rand(elty,m-1)
    dub = rand(elty,m-1)
    db = rand(elty,m)
    xa = rand(elty,m)
    xb = rand(elty,m)
    d_dl = CudaArray(vcat([0],dla,[0],dlb))
    d_du = CudaArray(vcat(dua,[0],dub,[0]))
    d_d = CudaArray(vcat(da,db))
    d_x = CudaArray(vcat(xa,xb))
    d_y = CUSPARSE.gtsvStridedBatch!(d_dl,d_d,d_du,d_x,2,m)
    Ca = diagm(da,0) + diagm(dua,1) + diagm(dla,-1)
    Cb = diagm(db,0) + diagm(dub,1) + diagm(dlb,-1)
    h_y = to_host(d_y)
    @test_approx_eq(h_y[1:m], Ca\xa)
    @test_approx_eq(h_y[m+1:2*m], Cb\xb)
end
test_gtsvStridedBatch(Float32)
test_gtsvStridedBatch(Float64)
test_gtsvStridedBatch(Complex64)
test_gtsvStridedBatch(Complex128)
