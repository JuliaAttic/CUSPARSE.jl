using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

##############
# test_csrmm #
##############

for (func,funcs,funch,func2,typ) in ((:test_csrmm,:test_csrmm_symm,:test_csrmm_herm,:test_csrmm2,:CudaSparseMatrixCSR),
                                     (:test_cscmm,:test_cscmm_symm,:test_cscmm_herm,:test_cscmm2,:CudaSparseMatrixCSC))
    @eval begin
        function $funcs(elty)
            A = sparse(rand(elty,m,m))
            A = A + A.'
            B = rand(elty,m,n)
            C = rand(elty,m,n)
            alpha = rand(elty)
            beta = rand(elty)
            d_B = CudaArray(B)
            d_C = CudaArray(C)
            d_A = Symmetric($typ(A))
            d_C = CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = to_host(d_C)
            C = alpha * A * B + beta * C
            @test_approx_eq(C,h_C)
            d_C = d_A.' * d_B
            h_C = to_host(d_C)
            C = A.' * B
            @test_approx_eq(C,h_C)
            d_B = CudaArray(rand(elty,k,n))
            @test_throws(DimensionMismatch, CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O'))
            @test_throws(DimensionMismatch, CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O'))
        end

        function $funch(elty)
            A = sparse(rand(elty,m,m))
            A = A + A'
            B = rand(elty,m,n)
            C = rand(elty,m,n)
            alpha = rand(elty)
            beta = rand(elty)
            d_B = CudaArray(B)
            d_C = CudaArray(C)
            d_A = Hermitian($typ(A))
            d_C = CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = to_host(d_C)
            C = alpha * A * B + beta * C
            @test_approx_eq(C,h_C)
            d_B = CudaArray(rand(elty,k,n))
            @test_throws(DimensionMismatch, CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O'))
            @test_throws(DimensionMismatch, CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O'))
        end

        function $func(elty)
            A = sparse(rand(elty,m,k))
            B = rand(elty,k,n)
            C = rand(elty,m,n)
            alpha = rand(elty)
            beta = rand(elty)
            d_B = CudaArray(B)
            d_C = CudaArray(C)
            d_A = $typ(A)
            @test_throws(DimensionMismatch, CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O'))
            @test_throws(DimensionMismatch, CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O'))
            d_D = CUSPARSE.mm('N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = to_host(d_D)
            D = alpha * A * B + beta * C
            @test_approx_eq(D,h_D)
            d_D = CUSPARSE.mm('N',d_A,d_B,beta,d_C,'O')
            h_D = to_host(d_D)
            D = A * B + beta * C
            @test_approx_eq(D,h_D)
            d_D = CUSPARSE.mm('N',d_A,d_B,d_C,'O')
            h_D = to_host(d_D)
            D = A * B + C
            @test_approx_eq(D,h_D)
            d_D = CUSPARSE.mm('N',alpha,d_A,d_B,'O')
            h_D = to_host(d_D)
            D = alpha * A * B
            @test_approx_eq(D,h_D)
            d_D = CUSPARSE.mm('N',d_A,d_B,'O')
            h_D = to_host(d_D)
            D = A * B
            @test_approx_eq(D,h_D)
            d_D = d_A*d_B
            h_D = to_host(d_D)
            D = A * B
            @test_approx_eq(D,h_D)
            @test_throws(DimensionMismatch, CUSPARSE.mm('T',alpha,d_A,d_B,beta,d_C,'O'))
            @test_throws(DimensionMismatch, CUSPARSE.mm('N',alpha,d_A,d_B,beta,d_B,'O'))
        end
        function $func2(elty)
            A = sparse(rand(elty,m,k))
            B = rand(elty,k,n)
            C = rand(elty,m,n)
            alpha = rand(elty)
            beta = rand(elty)
            d_B = CudaArray(B)
            d_C = CudaArray(C)
            d_A = $typ(A)
            @test_throws(DimensionMismatch, CUSPARSE.mm2!('N','T',alpha,d_A,d_B,beta,d_C,'O'))
            @test_throws(DimensionMismatch, CUSPARSE.mm2!('T','N',alpha,d_A,d_B,beta,d_C,'O'))
            @test_throws(DimensionMismatch, CUSPARSE.mm2!('T','T',alpha,d_A,d_B,beta,d_C,'O'))
            @test_throws(DimensionMismatch, CUSPARSE.mm2!('N','N',alpha,d_A,d_B,beta,d_B,'O'))
            d_D = CUSPARSE.mm2('N','N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = to_host(d_D)
            D = alpha * A * B + beta * C
            @test_approx_eq(D,h_D)
            d_D = CUSPARSE.mm2('N','N',d_A,d_B,beta,d_C,'O')
            h_D = to_host(d_D)
            D = A * B + beta * C
            @test_approx_eq(D,h_D)
            d_D = CUSPARSE.mm2('N','N',d_A,d_B,d_C,'O')
            h_D = to_host(d_D)
            D = A * B + C
            @test_approx_eq(D,h_D)
            d_D = CUSPARSE.mm2('N','N',alpha,d_A,d_B,'O')
            h_D = to_host(d_D)
            D = alpha * A * B
            @test_approx_eq(D,h_D)
            d_D = CUSPARSE.mm2('N','N',d_A,d_B,'O')
            h_D = to_host(d_D)
            D = A * B
            @test_approx_eq(D,h_D)
        end
    end
end

function test_bsrmm2(elty)
    A = sparse(rand(elty,m,k))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta = rand(elty)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
    @test_throws(DimensionMismatch, CUSPARSE.mm2('N','T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.mm2('T','N',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.mm2('T','T',alpha,d_A,d_B,beta,d_C,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.mm2('N','N',alpha,d_A,d_B,beta,d_B,'O'))
    d_D = CUSPARSE.mm2('N','N',alpha,d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = alpha * A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.mm2('N','N',d_A,d_B,beta,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + beta * C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.mm2('N','N',d_A,d_B,d_C,'O')
    h_D = to_host(d_D)
    D = A * B + C
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.mm2('N','N',alpha,d_A,d_B,'O')
    h_D = to_host(d_D)
    D = alpha * A * B
    @test_approx_eq(D,h_D)
    d_D = CUSPARSE.mm2('N','N',d_A,d_B,'O')
    h_D = to_host(d_D)
    D = A * B
    @test_approx_eq(D,h_D)
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_csrmm_symm(elty)
    test_cscmm_symm(elty)
    println("mm_symm! took ", toq(), " for ", elty)
    tic()
    test_csrmm_herm(elty)
    test_cscmm_herm(elty)
    println("mm_herm! took ", toq(), " for ", elty)
    tic()
    test_csrmm(elty)
    test_cscmm(elty)
    println("mm took ", toq(), " for ", elty)
    tic()
    test_csrmm2(elty)
    test_cscmm2(elty)
    test_bsrmm2(elty)
    println("mm2 took ", toq(), " for ", elty)
end
