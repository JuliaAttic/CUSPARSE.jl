using CUSPARSE
using CuArrays
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "cssv" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        @testset "csrsv" begin
            A = rand(elty,m,m)
            A = triu(A)
            x = rand(elty,m)
            alpha = rand(elty)
            d_x = CuArray(x)
            d_A = CudaSparseMatrixCSR(sparse(A))
            d_y = CUSPARSE.sv('N','T','U',alpha,d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\(alpha * x)
            @test y ≈ h_y
            x = rand(elty,n)
            d_x = CuArray(x)
            info = CUSPARSE.sv_analysis('N','T','U',d_A,'O')
            @test_throws DimensionMismatch CUSPARSE.sv_solve('N','U',alpha,d_A,d_x,info,'O')
            A = sparse(rand(elty,m,n))
            d_A = CudaSparseMatrixCSR(A)
            @test_throws DimensionMismatch CUSPARSE.sv_analysis('T','T','U',d_A,'O')
            CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
        end

        @testset "cscsv" begin
            A = rand(elty,m,m)
            A = triu(A)
            x = rand(elty,m)
            alpha = rand(elty)
            d_x = CuArray(x)
            d_A = CudaSparseMatrixCSC(sparse(A))
            d_y = CUSPARSE.sv('N','T','U',alpha,d_A,d_x,'O')
            h_y = collect(d_y)
            y = A\(alpha * x)
            @test y ≈ h_y
            x = rand(elty,n)
            d_x = CuArray(x)
            info = CUSPARSE.sv_analysis('N','T','U',d_A,'O')
            @test_throws DimensionMismatch CUSPARSE.sv_solve('N','U',alpha,d_A,d_x,info,'O')
            A = sparse(rand(elty,m,n))
            d_A = CudaSparseMatrixCSC(A)
            @test_throws DimensionMismatch CUSPARSE.sv_analysis('T','T','U',d_A,'O')
            CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
        end

        @testset "csrsv2" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CudaSparseMatrixCSR(sparse(A))
            d_Y = CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
            h_Y = collect(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_y = UpperTriangular(d_A)\d_X
            h_y = collect(d_y)
            y = A\X
            @test y ≈ h_y
            A = sparse(rand(elty,m,n))
            d_A = CudaSparseMatrixCSR(A)
            @test_throws DimensionMismatch CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
        end

        @testset "cscsv2" begin
            A = rand(elty,m,m)
            A = triu(A)
            X = rand(elty,m)
            alpha = rand(elty)
            d_X = CuArray(X)
            d_A = CudaSparseMatrixCSC(sparse(A))
            d_Y = CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
            h_Y = collect(d_Y)
            Y = A\(alpha * X)
            @test Y ≈ h_Y
            d_y = UpperTriangular(d_A)\d_X
            h_y = collect(d_y)
            y = A\X
            @test y ≈ h_y
            A = sparse(rand(elty,m,n))
            d_A = CudaSparseMatrixCSC(A)
            @test_throws DimensionMismatch CUSPARSE.sv2('N','U',alpha,d_A,d_X,'O')
        end
    end
end
