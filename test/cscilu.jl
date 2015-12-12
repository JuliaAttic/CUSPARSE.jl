using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

################
# test_cscilu0 #
################

function test_cscilu0!(elty)
    A = sparse(rand(elty,m,m))
    A = A + transpose(A)
    d_A = CudaSparseMatrixCSC(A)
    info = CUSPARSE.sv_analysis('N','S','U',d_A,'O')
    d_A = CUSPARSE.ilu0!('N',d_A,info,'O')
    h_A = to_host(d_A)
    Alu = lufact(full(A),Val{false})
    Ac = sparse(Alu[:L]*Alu[:U])
    h_A = ctranspose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
end

function test_cscilu0(elty)
    A = sparse(rand(elty,m,m))
    A = A + transpose(A)
    d_A = CudaSparseMatrixCSC(A)
    info = CUSPARSE.sv_analysis('N','S','U',d_A,'O')
    d_B = CUSPARSE.ilu0('N',d_A,info,'O')
    h_B = to_host(d_B)
    Alu = lufact(full(A),Val{false})
    Ac = sparse(Alu[:L]*Alu[:U])
    h_A = ctranspose(h_B) * h_B
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
end

#################
# test_cscilu02 #
#################

function test_cscilu02!(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSC(sparse(A))
    d_A = CUSPARSE.ilu02!(d_A,'O')
    h_A = to_host(d_A)
    Alu = lufact(full(A),Val{false})
    Ac = sparse(Alu[:L]*Alu[:U])
    h_A = ctranspose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
    d_A = CudaSparseMatrixCSC(sparse(rand(elty,m,n)))
    @test_throws(DimensionMismatch, CUSPARSE.ilu02!(d_A,'O'))
end

function test_cscilu02(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSC(sparse(A))
    d_B = CUSPARSE.ilu02(d_A,'O')
    h_A = to_host(d_B)
    Alu = lufact(full(A),Val{false})
    Ac = sparse(Alu[:L]*Alu[:U])
    h_A = ctranspose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_cscilu0!(elty)
    println("cscilu0! took ", toq(), " for ", elty)
    tic()
    test_cscilu0(elty)
    println("cscilu0 took ", toq(), " for ", elty)
    tic()
    test_cscilu02!(elty)
    println("cscilu02! took ", toq(), " for ", elty)
    tic()
    test_cscilu02(elty)
    println("cscilu02 took ", toq(), " for ", elty)
end
