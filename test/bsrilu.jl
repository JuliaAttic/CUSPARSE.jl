using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

#################
# test_bsrilu02 #
#################

function test_bsrilu02!(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    d_A = CUSPARSE.bsrilu02!(d_A,'O')
    h_A = to_host(CUSPARSE.switch2csr(d_A))
    Alu = lufact(full(A),Val{false})
    Ac = sparse(Alu[:L]*Alu[:U])
    h_A = ctranspose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
    d_A = CudaSparseMatrixCSR(sparse(rand(elty,m,n)))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    @test_throws(DimensionMismatch,CUSPARSE.bsrilu02!(d_A,'O'))
end
test_bsrilu02!(Float32)
test_bsrilu02!(Float64)
test_bsrilu02!(Complex64)
test_bsrilu02!(Complex128)

function test_bsrilu02(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(A))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    d_B = CUSPARSE.bsrilu02(d_A,'O')
    h_A = to_host(CUSPARSE.switch2csr(d_B))
    Alu = lufact(full(A),Val{false})
    Ac = sparse(Alu[:L]*Alu[:U])
    h_A = ctranspose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
end
test_bsrilu02(Float32)
test_bsrilu02(Float64)
test_bsrilu02(Complex64)
test_bsrilu02(Complex128)

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_bsrilu02!(elty)
    println("bsrilu02! took ", toq(), " for ", elty)
    tic()
    test_bsrilu02(elty)
    println("bsrilu02 took ", toq(), " for ", elty)
end
