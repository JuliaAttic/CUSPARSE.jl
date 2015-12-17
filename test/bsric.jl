using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

################
# test_bsric02 #
################

function test_bsric02!(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(tril(A)))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    d_A = CUSPARSE.ic02!(d_A,'O')
    h_A = to_host(CUSPARSE.switch2csr(d_A))
    Ac = sparse(full(cholfact(A)))
    h_A = transpose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
    d_A = CudaSparseMatrixCSR(sparse(tril(rand(elty,m,n))))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    @test_throws(DimensionMismatch,CUSPARSE.ic02!(d_A,'O'))
end

function test_bsric02(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSR(sparse(tril(A)))
    d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
    d_B = CUSPARSE.ic02(d_A,'O')
    h_A = to_host(CUSPARSE.switch2csr(d_B))
    Ac = sparse(full(cholfact(A)))
    h_A = transpose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_bsric02!(elty)
    println("bsric02! took ", toq(), " for ", elty)
    tic()
    test_bsric02(elty)
    println("bsric02 took ", toq(), " for ", elty)
end
