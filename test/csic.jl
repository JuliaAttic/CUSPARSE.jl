using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

for (funcr,funcc,func2,typ) in ((:test_csric0_real,:test_csric0_complex,:test_csric02,:CudaSparseMatrixCSR),
                                (:test_cscic0_real,:test_cscic0_complex,:test_cscic02,:CudaSparseMatrixCSC))
    @eval begin
        function $funcr(elty)
            A = rand(elty,m,m)
            A = A + transpose(A)
            A += m * eye(elty,m)
            d_A = Symmetric($typ(sparse(triu(A))))
            info = CUSPARSE.sv_analysis('N','S','U',d_A,'O')
            d_B = CUSPARSE.ic0('N','S',d_A,info,'O')
            h_A = to_host(d_B)
            Ac = sparse(full(cholfact(A)))
            h_A = transpose(h_A) * h_A
            @test_approx_eq(h_A.rowval,Ac.rowval)
            @test reduce(&,isfinite(h_A.nzval))
        end

        function $funcc(elty)
            A = rand(elty,m,m)
            A = A + ctranspose(A)
            A += m * eye(elty,m)
            d_A = Hermitian($typ(sparse(triu(A))))
            info = CUSPARSE.sv_analysis('N','H','U',d_A,'O')
            d_B = CUSPARSE.ic0('N','H',d_A,info,'O')
            h_A = to_host(d_B)
            Ac = sparse(full(cholfact(A)))
            h_A = ctranspose(h_A) * h_A
            @test_approx_eq(h_A.rowval,Ac.rowval)
            @test reduce(&,isfinite(h_A.nzval))
        end

        function $func2(elty)
            A = rand(elty,m,m)
            A += transpose(A)
            A += m * eye(elty,m)
            d_A = $typ(sparse(tril(A)))
            d_B = CUSPARSE.ic02(d_A,'O')
            h_A = to_host(d_B)
            Ac = sparse(full(cholfact(A)))
            h_A = transpose(h_A) * h_A
            @test_approx_eq(h_A.rowval,Ac.rowval)
            @test reduce(&,isfinite(h_A.nzval))
            A = rand(elty,m,n)
            d_A = $typ(sparse(tril(A)))
            @test_throws(DimensionMismatch,CUSPARSE.ic02(d_A,'O'))
        end
    end
end

types = [Float32,Float64]
for elty in types
    tic()
    test_csric0_real(elty)
    test_cscic0_real(elty)
    println("ic0 took ", toq(), " for ", elty)
    tic()
    test_csric02(elty)
    test_cscic02(elty)
    println("ic02 took ", toq(), " for ", elty)
end

types = [Complex64,Complex128]
for elty in types
    tic()
    test_csric0_complex(elty)
    test_cscic0_complex(elty)
    println("ic0 took ", toq(), " for ", elty)
    tic()
    test_csric02(elty)
    test_cscic02(elty)
    println("ic02 took ", toq(), " for ", elty)
end
