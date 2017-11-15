using CUSPARSE
using CUDAdrv
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "mv" begin
    for elty in [Float32,Float64,Complex64,Complex128]
        A = sparse(rand(elty,m,m))
        x = rand(elty,m)
        y = rand(elty,m)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "mv_symm" begin
            A_s = A + A.'
            @testset "csr" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = Symmetric(CudaSparseMatrixCSR(A_s))
                d_y = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_y = to_host(d_y)
                z = alpha * A_s * x + beta * y
                @test z ≈ h_y
                x_  = rand(elty,n)
                d_x = CuArray(x_)
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
            end
            @testset "csc" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = Symmetric(CudaSparseMatrixCSC(A_s))
                d_y = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_y = to_host(d_y)
                z = alpha * A_s * x + beta * y
                @test z ≈ h_y
                x_  = rand(elty,n)
                d_x = CuArray(x_)
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
            end
        end

        @testset "mv_herm" begin
            A_h = A + A'
            @testset "csr" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = Hermitian(CudaSparseMatrixCSR(A_h))
                d_y = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_y = to_host(d_y)
                z = alpha * A_h * x + beta * y
                @test z ≈ h_y
                x_  = rand(elty,n)
                d_x = CuArray(x_)
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
            end
            @testset "csc" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = Hermitian(CudaSparseMatrixCSC(A_h))
                d_y = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_y = to_host(d_y)
                z = alpha * A_h * x + beta * y
                @test z ≈ h_y
                x_  = rand(elty,n)
                d_x = CuArray(x_)
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
            end
        end
        A = sparse(rand(elty,m,n))
        x = rand(elty,n)
        y = rand(elty,m)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "mv" begin
            @testset "csr" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = CudaSparseMatrixCSR(A)
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_z = to_host(d_z)
                z = alpha * A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,beta,d_y,'O')
                h_z = to_host(d_z)
                z = A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,d_y,'O')
                h_z = to_host(d_z)
                z = A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,d_y,'O')
                h_z = to_host(d_z)
                z = alpha * A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,'O')
                h_z = to_host(d_z)
                z = alpha * A * x
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,'O')
                h_z = to_host(d_z)
                z = A * x
                @test z ≈ h_z
                d_z = d_A*d_x
                h_z = to_host(d_z)
                z = A * x
                @test z ≈ h_z
            end
            @testset "csc" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = CudaSparseMatrixCSC(A)
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_z = to_host(d_z)
                z = alpha * A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,beta,d_y,'O')
                h_z = to_host(d_z)
                z = A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,d_y,'O')
                h_z = to_host(d_z)
                z = A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,d_y,'O')
                h_z = to_host(d_z)
                z = alpha * A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,'O')
                h_z = to_host(d_z)
                z = alpha * A * x
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,'O')
                h_z = to_host(d_z)
                z = A * x
                @test z ≈ h_z
                d_z = d_A*d_x
                h_z = to_host(d_z)
                z = A * x
                @test z ≈ h_z
            end
            @testset "bsr" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = CudaSparseMatrixCSR(A)
                d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_z = to_host(d_z)
                z = alpha * A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,beta,d_y,'O')
                h_z = to_host(d_z)
                z = A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,d_y,'O')
                h_z = to_host(d_z)
                z = A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,d_y,'O')
                h_z = to_host(d_z)
                z = alpha * A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,'O')
                h_z = to_host(d_z)
                z = alpha * A * x
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,'O')
                h_z = to_host(d_z)
                z = A * x
                @test z ≈ h_z
                d_z = d_A*d_x
                h_z = to_host(d_z)
                z = A * x
                @test z ≈ h_z
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
            end
            @testset "hyb" begin
                d_x = CuArray(x)
                d_y = CuArray(y)
                d_A = CudaSparseMatrixCSR(A)
                d_A = CUSPARSE.switch2hyb(d_A)
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,beta,d_y,'O')
                h_z = to_host(d_z)
                z = alpha * A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,beta,d_y,'O')
                h_z = to_host(d_z)
                z = A * x + beta * y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,d_y,'O')
                h_z = to_host(d_z)
                z = A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,d_y,'O')
                h_z = to_host(d_z)
                z = alpha * A * x + y
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',alpha,d_A,d_x,'O')
                h_z = to_host(d_z)
                z = alpha * A * x
                @test z ≈ h_z
                d_z = CUSPARSE.mv('N',d_A,d_x,'O')
                h_z = to_host(d_z)
                z = A * x
                @test z ≈ h_z
                d_z = d_A*d_x
                h_z = to_host(d_z)
                z = A * x
                @test z ≈ h_z
                @test_throws DimensionMismatch CUSPARSE.mv('T',alpha,d_A,d_x,beta,d_y,'O')
                @test_throws DimensionMismatch CUSPARSE.mv('N',alpha,d_A,d_y,beta,d_x,'O')
            end
        end
    end
end
