using CUSPARSE
using CUDAdrv
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "mm" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        A = sparse(rand(elty,m,k))
        B = rand(elty,k,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = CudaSparseMatrixCSR(A)
            @test_throws DimensionMismatch CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O')
            d_D = CUSPARSE.mm('N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = to_host(d_D)
            D = alpha * A * B + beta * C
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',d_A,d_B,beta,d_C,'O')
            h_D = to_host(d_D)
            D = A * B + beta * C
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',d_A,d_B,d_C,'O')
            h_D = to_host(d_D)
            D = A * B + C
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',alpha,d_A,d_B,'O')
            h_D = to_host(d_D)
            D = alpha * A * B
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',d_A,d_B,'O')
            h_D = to_host(d_D)
            D = A * B
            @test D ≈ h_D
            d_D = d_A*d_B
            h_D = to_host(d_D)
            D = A * B
            @test D ≈ h_D
            @test_throws DimensionMismatch CUSPARSE.mm('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm('N',alpha,d_A,d_B,beta,d_B,'O')
        end
        @testset "csc" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = CudaSparseMatrixCSC(A)
            @test_throws DimensionMismatch CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O')
            d_D = CUSPARSE.mm('N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = to_host(d_D)
            D = alpha * A * B + beta * C
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',d_A,d_B,beta,d_C,'O')
            h_D = to_host(d_D)
            D = A * B + beta * C
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',d_A,d_B,d_C,'O')
            h_D = to_host(d_D)
            D = A * B + C
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',alpha,d_A,d_B,'O')
            h_D = to_host(d_D)
            D = alpha * A * B
            @test D ≈ h_D
            d_D = CUSPARSE.mm('N',d_A,d_B,'O')
            h_D = to_host(d_D)
            D = A * B
            @test D ≈ h_D
            d_D = d_A*d_B
            h_D = to_host(d_D)
            D = A * B
            @test D ≈ h_D
            @test_throws DimensionMismatch CUSPARSE.mm('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm('N',alpha,d_A,d_B,beta,d_B,'O')
        end
    end
end

@testset "mm_symm" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        A = sparse(rand(elty,m,m))
        A = A + A.'
        B = rand(elty,m,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = Symmetric(CudaSparseMatrixCSR(A))
            d_C = CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = to_host(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_C
            d_C = d_A.' * d_B
            h_C = to_host(d_C)
            D = A.' * B
            @test D ≈ h_C
            d_B = CuArray(rand(elty,k,n))
            @test_throws DimensionMismatch CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O')
        end
        @testset "csc" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = Symmetric(CudaSparseMatrixCSC(A))
            d_C = CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = to_host(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_C
            d_C = d_A.' * d_B
            h_C = to_host(d_C)
            D = A.' * B
            @test D ≈ h_C
            d_B = CuArray(rand(elty,k,n))
            @test_throws DimensionMismatch CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O')
        end
    end
end
@testset "mm_herm" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        A = sparse(rand(elty,m,m))
        A = A + A'
        B = rand(elty,m,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = Hermitian(CudaSparseMatrixCSR(A))
            d_C = CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = to_host(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_C
            d_B = CuArray(rand(elty,k,n))
            @test_throws DimensionMismatch CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O')
        end
        @testset "csc" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = Hermitian(CudaSparseMatrixCSC(A))
            d_C = CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_C,'O')
            h_C = to_host(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_C
            d_B = CuArray(rand(elty,k,n))
            @test_throws DimensionMismatch CUSPARSE.mm!('T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm!('N',alpha,d_A,d_B,beta,d_B,'O')
        end
    end
end

@testset "mm2" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        A = sparse(rand(elty,m,k))
        B = rand(elty,k,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = CudaSparseMatrixCSR(A)
            @test_throws DimensionMismatch CUSPARSE.mm2!('N','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm2!('T','N',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm2!('T','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm2!('N','N',alpha,d_A,d_B,beta,d_B,'O')
            d_D = CUSPARSE.mm2('N','N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = to_host(d_D)
            D = alpha * A * B + beta * C
            @test D ≈ h_D
            d_D = CUSPARSE.mm2('N','N',d_A,d_B,beta,d_C,'O')
            h_D = to_host(d_D)
            D = A * B + beta * C
            @test D ≈ h_D
            d_D = CUSPARSE.mm2('N','N',d_A,d_B,d_C,'O')
            h_D = to_host(d_D)
            D = A * B + C
            @test D ≈ h_D
            d_D = CUSPARSE.mm2('N','N',alpha,d_A,d_B,'O')
            h_D = to_host(d_D)
            D = alpha * A * B
            @test D ≈ h_D
            d_D = CUSPARSE.mm2('N','N',d_A,d_B,'O')
            h_D = to_host(d_D)
            D = A * B
            @test D ≈ h_D
        end
        @testset "csc" begin
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_A = CudaSparseMatrixCSC(A)
            @test_throws DimensionMismatch CUSPARSE.mm2!('N','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm2!('T','N',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm2!('T','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch CUSPARSE.mm2!('N','N',alpha,d_A,d_B,beta,d_B,'O')
            d_D = CUSPARSE.mm2('N','N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = to_host(d_D)
            D = alpha * A * B + beta * C
            @test D ≈ h_D
            d_D = CUSPARSE.mm2('N','N',d_A,d_B,beta,d_C,'O')
            h_D = to_host(d_D)
            D = A * B + beta * C
            @test D ≈ h_D
            d_D = CUSPARSE.mm2('N','N',d_A,d_B,d_C,'O')
            h_D = to_host(d_D)
            D = A * B + C
            @test D ≈ h_D
            d_D = CUSPARSE.mm2('N','N',alpha,d_A,d_B,'O')
            h_D = to_host(d_D)
            D = alpha * A * B
            @test D ≈ h_D
            d_D = CUSPARSE.mm2('N','N',d_A,d_B,'O')
            h_D = to_host(d_D)
            D = A * B
            @test D ≈ h_D
        end
    end
end
@testset "bsrmm2" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        A = sparse(rand(elty,m,k))
        B = rand(elty,k,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        d_B = CuArray(B)
        d_C = CuArray(C)
        d_A = CudaSparseMatrixCSR(A)
        d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,blockdim))
        @test_throws DimensionMismatch CUSPARSE.mm2('N','T',alpha,d_A,d_B,beta,d_C,'O')
        @test_throws DimensionMismatch CUSPARSE.mm2('T','N',alpha,d_A,d_B,beta,d_C,'O')
        @test_throws DimensionMismatch CUSPARSE.mm2('T','T',alpha,d_A,d_B,beta,d_C,'O')
        @test_throws DimensionMismatch CUSPARSE.mm2('N','N',alpha,d_A,d_B,beta,d_B,'O')
        d_D = CUSPARSE.mm2('N','N',alpha,d_A,d_B,beta,d_C,'O')
        h_D = to_host(d_D)
        D = alpha * A * B + beta * C
        @test D ≈ h_D
        d_D = CUSPARSE.mm2('N','N',d_A,d_B,beta,d_C,'O')
        h_D = to_host(d_D)
        D = A * B + beta * C
        @test D ≈ h_D
        d_D = CUSPARSE.mm2('N','N',d_A,d_B,d_C,'O')
        h_D = to_host(d_D)
        D = A * B + C
        @test D ≈ h_D
        d_D = CUSPARSE.mm2('N','N',alpha,d_A,d_B,'O')
        h_D = to_host(d_D)
        D = alpha * A * B
        @test D ≈ h_D
        d_D = CUSPARSE.mm2('N','N',d_A,d_B,'O')
        h_D = to_host(d_D)
        D = A * B
        @test D ≈ h_D
    end
end
