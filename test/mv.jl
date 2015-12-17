using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

for (func,funcs,funch,typ) in ((:test_csrmv,:test_csrmv_symm,:test_csrmv_herm,:CudaSparseMatrixCSR),
                               (:test_cscmv,:test_cscmv_symm,:test_cscmv_herm,:CudaSparseMatrixCSC))
    @eval begin
        function $funcs(elty)
            A = sparse(rand(elty,m,m))
            A = A + A.'
            x = rand(elty,m)
            y = rand(elty,m)
            alpha = rand(elty)
            beta = rand(elty)
            d_x = CudaArray(x)
            d_y = CudaArray(y)
            d_A = Symmetric($typ(A))
            d_y = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
            h_y = to_host(d_y)
            y = alpha * A * x + beta * y
            @test_approx_eq(y,h_y)
            x   = rand(elty,n)
            d_x = CudaArray(x)
            @test_throws(DimensionMismatch, CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O'))
            @test_throws(DimensionMismatch, CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O'))
        end

        function $funch(elty)
            A = sparse(rand(elty,m,m))
            A = A + A'
            x = rand(elty,m)
            y = rand(elty,m)
            alpha = rand(elty)
            beta = rand(elty)
            d_x = CudaArray(x)
            d_y = CudaArray(y)
            d_A = Hermitian($typ(A))
            d_y = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
            h_y = to_host(d_y)
            y = alpha * A * x + beta * y
            @test_approx_eq(y,h_y)
            x   = rand(elty,n)
            d_x = CudaArray(x)
            @test_throws(DimensionMismatch, CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O'))
            @test_throws(DimensionMismatch, CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O'))
        end

        function $func(elty)
            A = sparse(rand(elty,m,n))
            x = rand(elty,n)
            y = rand(elty,m)
            alpha = rand(elty)
            beta = rand(elty)
            d_x = CudaArray(x)
            d_y = CudaArray(y)
            d_A = $typ(A)
            @test_throws(DimensionMismatch, CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O'))
            @test_throws(DimensionMismatch, CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O'))
            d_z = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
            h_z = to_host(d_z)
            z = alpha * A * x + beta * y
            @test_approx_eq(z,h_z)
            d_z = CUSPARSE.mv('N',d_A,d_x,beta,d_y,'O')
            h_z = to_host(d_z)
            z = A * x + beta * y
            @test_approx_eq(z,h_z)
            d_z = CUSPARSE.mv('N',d_A,d_x,d_y,'O')
            h_z = to_host(d_z)
            z = A * x + y
            @test_approx_eq(z,h_z)
            d_z = CUSPARSE.mv('N',alpha,d_A,d_x,d_y,'O')
            h_z = to_host(d_z)
            z = alpha * A * x + y
            @test_approx_eq(z,h_z)
            d_z = CUSPARSE.mv('N',alpha,d_A,d_x,'O')
            h_z = to_host(d_z)
            z = alpha * A * x
            @test_approx_eq(z,h_z)
            d_z = CUSPARSE.mv('N',d_A,d_x,'O')
            h_z = to_host(d_z)
            z = A * x
            @test_approx_eq(z,h_z)
            d_z = d_A*d_x
            h_z = to_host(d_z)
            z = A * x
            @test_approx_eq(z,h_z)
        end
    end
end

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
    @test_throws(DimensionMismatch, CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O'))
    d_z = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',alpha,d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',alpha,d_A,d_x,'O')
    h_z = to_host(d_z)
    z = alpha * A * x
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',d_A,d_x,'O')
    h_z = to_host(d_z)
    z = A * x
    @test_approx_eq(z,h_z)
    d_z = d_A*d_x
    h_z = to_host(d_z)
    z = A * x
    @test_approx_eq(z,h_z)
    @test_throws(DimensionMismatch, CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O'))
end

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
    d_z = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',d_A,d_x,beta,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + beta * y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',alpha,d_A,d_x,d_y,'O')
    h_z = to_host(d_z)
    z = alpha * A * x + y
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',alpha,d_A,d_x,'O')
    h_z = to_host(d_z)
    z = alpha * A * x
    @test_approx_eq(z,h_z)
    d_z = CUSPARSE.mv('N',d_A,d_x,'O')
    h_z = to_host(d_z)
    z = A * x
    @test_approx_eq(z,h_z)
    d_z = d_A*d_x
    h_z = to_host(d_z)
    z = A * x
    @test_approx_eq(z,h_z)
    @test_throws(DimensionMismatch, CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O'))
    @test_throws(DimensionMismatch, CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O'))
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_csrmv_symm(elty)
    test_cscmv_symm(elty)
    println("mv_symm took ", toq(), " for ", elty)
    tic()
    test_csrmv_herm(elty)
    test_cscmv_herm(elty)
    println("mv_herm took ", toq(), " for ", elty)
    tic()
    test_csrmv(elty)
    test_cscmv(elty)
    test_bsrmv(elty)
    test_hybmv(elty)
    println("mv took ", toq(), " for ", elty)
end
