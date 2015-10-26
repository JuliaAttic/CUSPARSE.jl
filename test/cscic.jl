using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

###############
# test_cscic0 #
###############

function test_cscic0_real!(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSC(sparse(triu(A)))
    info = CUSPARSE.cscsv_analysis('N','S','U',d_A,'O')
    d_A = CUSPARSE.cscic0!('N','S',d_A,info,'O')
    h_A = to_host(d_A)
    Ac = sparse(full(cholfact(A)))
    h_A = transpose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
end

function test_cscic0_real(elty)
    A = rand(elty,m,m)
    A = A + transpose(A) 
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSC(sparse(triu(A)))
    info = CUSPARSE.cscsv_analysis('N','S','U',d_A,'O')
    d_B = CUSPARSE.cscic0('N','S',d_A,info,'O')
    h_A = to_host(d_B)
    Ac = sparse(full(cholfact(A)))
    h_A = transpose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
end

function test_cscic0_complex!(elty)
    A = rand(elty,m,m)
    A = A + ctranspose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSC(sparse(triu(A)))
    info = CUSPARSE.cscsv_analysis('N','H','U',d_A,'O')
    d_A = CUSPARSE.cscic0!('N','H',d_A,info,'O')
    h_A = to_host(d_A)
    Ac = sparse(full(cholfact(A)))
    h_A = ctranspose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
end

function test_cscic0_complex(elty)
    A = rand(elty,m,m)
    A = A + ctranspose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSC(sparse(triu(A)))
    info = CUSPARSE.cscsv_analysis('N','H','U',d_A,'O')
    d_B = CUSPARSE.cscic0('N','H',d_A,info,'O')
    h_A = to_host(d_B)
    Ac = sparse(full(cholfact(A)))
    h_A = ctranspose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
end

################
# test_cscic02 #
################

function test_cscic02!(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSC(sparse(tril(A)))
    d_A = CUSPARSE.cscic02!(d_A,'O')
    h_A = to_host(d_A)
    Ac = sparse(full(cholfact(A)))
    h_A = transpose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
end

function test_cscic02(elty)
    A = rand(elty,m,m)
    A += transpose(A)
    A += m * eye(elty,m)
    d_A = CudaSparseMatrixCSC(sparse(tril(A)))
    d_B = CUSPARSE.cscic02(d_A,'O')
    h_A = to_host(d_B)
    Ac = sparse(full(cholfact(A)))
    h_A = transpose(h_A) * h_A
    @test_approx_eq(h_A.rowval,Ac.rowval)
    @test reduce(&,isfinite(h_A.nzval))
    A = rand(elty,m,n)
    d_A = CudaSparseMatrixCSC(sparse(tril(A)))
    @test_throws(DimensionMismatch,CUSPARSE.cscic02(d_A,'O'))
end

types = [Float32,Float64]
for elty in types
    tic()
    test_cscic0_real!(elty)
    println("cscic0! took ", toq(), " for ", elty)
    tic()
    test_cscic0_real(elty)
    println("cscic0 took ", toq(), " for ", elty)
    tic()
    test_cscic02!(elty)
    println("cscic02! took ", toq(), " for ", elty)
    tic()
    test_cscic02(elty)
    println("cscic02 took ", toq(), " for ", elty)
end
