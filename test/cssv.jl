using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

##############
# test_csrsv #
##############

for (func,func2,typ) in ((:test_csrsv,:test_csrsv2,:CudaSparseMatrixCSR),
                         (:test_cscsv,:test_cscsv2,:CudaSparseMatrixCSC))
    @eval begin
        function $func(elty)
            A = rand(elty,m,m)
            A = triu(A)
            x = rand(elty,m)
            alpha = rand(elty)
            d_x = CudaArray(x)
            d_A = $typ(sparse(A))
            d_y = CUSPARSE.sv('N','T','U',alpha,d_A,d_x,'O')
            h_y = to_host(d_y)
            y = A\(alpha * x)
            @test_approx_eq(y,h_y)
            x = rand(elty,n)
            d_x = CudaArray(x)
            info = CUSPARSE.sv_analysis('N','T','U',d_A,'O')
            @test_throws(DimensionMismatch, CUSPARSE.sv_solve('N','U',alpha,d_A,d_x,info,'O'))
            A = sparse(rand(elty,m,n))
            d_A = $typ(A)
            @test_throws(DimensionMismatch, CUSPARSE.sv_analysis('T','T','U',d_A,'O'))
            CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
        end

        function $func2(elty)
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m)
            alpha = rand(elty)
            d_X = CudaArray(X)
            d_A = $typ(sparse(A))
            d_Y = CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
            h_Y = to_host(d_Y)
            Y = A\(alpha * X)
            @test_approx_eq(Y,h_Y)
            d_y = UpperTriangular(d_A)\d_X
            h_y = to_host(d_y)
            y = A\X
            @test_approx_eq(y,h_y)
            A = sparse(rand(elty,m,n))
            d_A = $typ(A)
            @test_throws(DimensionMismatch, CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O'))
        end
    end
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_csrsv(elty)
    test_cscsv(elty)
    println("cssv took ", toq(), " for ", elty)
    tic()
    test_csrsv2(elty)
    test_cscsv2(elty)
    println("cssv2 took ", toq(), " for ", elty)
end
