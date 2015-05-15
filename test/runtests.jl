using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

# misc util tests
x = sprand(m,n,0.2)
d_x = CudaSparseMatrixCSC(x)
@test device(x)   == -1
@test length(d_x) == m*n
@test size(d_x)   == (m,n)
@test size(d_x,1) == m
@test size(d_x,2) == n
y = sprand(k,n,0.2)
d_y = CudaSparseMatrixCSC(y)
@test_throws ArgumentError copy!(d_y,d_x)
d_y = CUSPARSE.switch2csr(d_y)
d_x = CUSPARSE.switch2csr(d_x)
@test_throws ArgumentError copy!(d_y,d_x)
d_y = CUSPARSE.switch2bsr(d_y,convert(Cint,blockdim))
d_x = CUSPARSE.switch2bsr(d_x,convert(Cint,blockdim))
@test_throws ArgumentError copy!(d_y,d_x)

# misc char tests

@test_throws ArgumentError CUSPARSE.cusparseop('Z')
@test_throws ArgumentError CUSPARSE.cusparsetype('Z')
@test_throws ArgumentError CUSPARSE.cusparsefill('Z')
@test_throws ArgumentError CUSPARSE.cusparsediag('Z')
@test_throws ArgumentError CUSPARSE.cusparsedir('Z')

# conversion
function test_make_csc(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSC(x)
    h_x = to_host(d_x)
    @test h_x == x
    @test eltype(d_x) == elty
end
test_make_csc(Float32)
test_make_csc(Float64)
test_make_csc(Complex64)
test_make_csc(Complex128)

function test_make_csr(elty)
    x = sparse(rand(elty,m,n))
    d_x = CudaSparseMatrixCSR(x)
    h_x = to_host(d_x)
    @test h_x == x
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

testnames = ["axpyi","dot","gthr","roti","sctr","csrsv","bsrsv","hybsv","csrmv","bsrmv","hybmv","csrmm","bsrmm","csrsm","bsrsm","gemm","geam","csric","csrilu","bsric","bsrilu","gtsv"]
chosentests = testnames
if( !isempty(ARGS) )
    chosentests = ARGS
end

for test in chosentests
    include("$test.jl")
end
