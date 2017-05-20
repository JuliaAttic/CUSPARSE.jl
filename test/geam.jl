using CUSPARSE
using CUDArt
using Base.Test

m = 5
n = 5
k = 10
blockdim = 5

@testset "geam" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        A = sparse(rand(elty,m,n))
        B = sparse(rand(elty,m,n))
        alpha = rand(elty)
        beta = rand(elty)
        @testset "csr" begin
            d_A = CudaSparseMatrixCSR(A)
            d_B = CudaSparseMatrixCSR(B)
            C = alpha * A + beta * B
            d_C = CUSPARSE.geam(alpha,d_A,beta,d_B,'O','O','O')
            h_C = to_host(d_C)
            @test C ≈ h_C
            d_C = CUSPARSE.geam(d_A,beta,d_B,'O','O','O')
            h_C = to_host(d_C)
            C = A + beta * B
            @test C ≈ h_C
            d_C = CUSPARSE.geam(alpha,d_A,d_B,'O','O','O')
            h_C = to_host(d_C)
            C = alpha * A + B
            @test C ≈ h_C
            d_C = CUSPARSE.geam(d_A,d_B,'O','O','O')
            h_C = to_host(d_C)
            C = A + B
            @test C ≈ h_C
            d_C = d_A + d_B
            h_C = to_host(d_C)
            C = A + B
            @test C ≈ h_C
            d_C = d_A - d_B
            h_C = to_host(d_C)
            C = A - B
            @test C ≈ h_C
            B_ = sparse(rand(elty,k,n))
            d_B = CudaSparseMatrixCSR(B_)
            @test_throws DimensionMismatch CUSPARSE.geam(d_B,d_A,'O','O','O')
        end
        @testset "csc" begin
            d_A = CudaSparseMatrixCSC(A)
            d_B = CudaSparseMatrixCSC(B)
            C = alpha * A + beta * B
            d_C = CUSPARSE.geam(alpha,d_A,beta,d_B,'O','O','O')
            h_C = to_host(d_C)
            @test C ≈ h_C
            d_C = CUSPARSE.geam(d_A,beta,d_B,'O','O','O')
            h_C = to_host(d_C)
            C = A + beta * B
            @test C ≈ h_C
            d_C = CUSPARSE.geam(alpha,d_A,d_B,'O','O','O')
            h_C = to_host(d_C)
            C = alpha * A + B
            @test C ≈ h_C
            d_C = CUSPARSE.geam(d_A,d_B,'O','O','O')
            h_C = to_host(d_C)
            C = A + B
            @test C ≈ h_C
            d_C = d_A + d_B
            h_C = to_host(d_C)
            C = A + B
            @test C ≈ h_C
            d_C = d_A - d_B
            h_C = to_host(d_C)
            C = A - B
            @test C ≈ h_C
            B_ = sparse(rand(elty,k,n))
            d_B = CudaSparseMatrixCSC(B_)
            @test_throws DimensionMismatch CUSPARSE.geam(d_B,d_A,'O','O','O')
        end
        @testset "mixed" begin
            A = spdiagm(rand(elty,m))
            B = spdiagm(rand(elty,m)) + sparse(diagm(rand(elty,m-1),1))
            d_A = CudaSparseMatrixCSR(A)
            d_B = CudaSparseMatrixCSC(B)
            d_C = d_B + d_A
            h_C = to_host(d_C)
            @test h_C ≈ A + B
            d_C = d_A + d_B
            h_C = to_host(d_C)
            @test h_C ≈ A + B
            d_C = d_A - d_B
            h_C = to_host(d_C)
            @test h_C ≈ A - B
            d_C = d_B - d_A
            h_C = to_host(d_C)
            @test h_C ≈ B - A
        end
    end
end
