using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

for (func,func2,typ) in ((:test_csrilu0,:test_csrilu02,:CudaSparseMatrixCSR),
                         (:test_cscilu0,:test_cscilu02,:CudaSparseMatrixCSC))
    @eval begin
        function $func(elty)
            A = rand(elty,m,m)
            d_A = $typ(sparse(A))
            info = CUSPARSE.sv_analysis('N','G','U',d_A,'O')
            d_B = CUSPARSE.ilu0('N',d_A,info,'O')
            h_B = to_host(d_B)
            Alu = lufact(full(A),Val{false})
            Ac = sparse(Alu[:L]*Alu[:U])
            h_A = ctranspose(h_B) * h_B
            @test_approx_eq(h_A.rowval,Ac.rowval)
            @test reduce(&,isfinite(h_A.nzval))
        end

        function $func2(elty)
            A = rand(elty,m,m)
            A += transpose(A)
            A += m * eye(elty,m)
            d_A = $typ(sparse(A))
            d_B = CUSPARSE.ilu02(d_A,'O')
            h_A = to_host(d_B)
            Alu = lufact(full(A),Val{false})
            Ac = sparse(Alu[:L]*Alu[:U])
            h_A = ctranspose(h_A) * h_A
            @test_approx_eq(h_A.rowval,Ac.rowval)
            @test reduce(&,isfinite(h_A.nzval))
        end
    end
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_csrilu0(elty)
    test_cscilu0(elty)
    println("ilu0 took ", toq(), " for ", elty)
    tic()
    test_csrilu02(elty)
    test_cscilu02(elty)
    println("ilu02 took ", toq(), " for ", elty)
end
