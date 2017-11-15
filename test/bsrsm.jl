using CUSPARSE
using CuArrays
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "bsrsm2" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        @testset "bsrsm2!" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m,n)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CudaSparseMatrixCSR(sparse(A))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_X = CUSPARSE.bsrsm2!('N','N',alpha,d_A,d_X,'O')
            h_Y = to_host(d_X)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_X = CuArray(rand(elty,n,n))
            @test_throws DimensionMismatch CUSPARSE.bsrsm2!('N','N',alpha,d_A,d_X,'O')
            @test_throws DimensionMismatch CUSPARSE.bsrsm2!('N','T',alpha,d_A,d_X,'O')
            A = sparse(rand(elty,m,n))
            d_A = CudaSparseMatrixCSR(A)
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            @test_throws DimensionMismatch CUSPARSE.bsrsm2!('N','N',alpha,d_A,d_X,'O')
        end

        @testset "bsrsm2" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m,n)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CudaSparseMatrixCSR(sparse(A))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_Y = CUSPARSE.bsrsm2('N','N',alpha,d_A,d_X,'O')
            h_Y = to_host(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            A = sparse(rand(elty,m,n))
            d_A = CudaSparseMatrixCSR(A)
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            @test_throws DimensionMismatch CUSPARSE.bsrsm2('N','N',alpha,d_A,d_X,'O')
        end
    end
end
