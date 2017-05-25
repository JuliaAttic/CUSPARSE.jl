using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5
@testset "ic0" begin
    @testset for elty in [Float32,Float64]
        @testset "csr" begin
            A    = rand(elty, m ,m)
            A    = A + transpose(A)
            A   += m * eye(elty ,m)
            d_A  = Symmetric(CudaSparseMatrixCSR(sparse(triu(A))))
            info = CUSPARSE.sv_analysis('N', 'S', 'U', d_A, 'O')
            d_B  = CUSPARSE.ic0('N', 'S', d_A, info, 'O')
            h_A  = to_host(d_B)
            Ac   = sparse(full(cholfact(A)))
            h_A  = transpose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
        @testset "csc" begin
            A    = rand(elty, m ,m)
            A    = A + transpose(A)
            A   += m * eye(elty ,m)
            d_A  = Symmetric(CudaSparseMatrixCSC(sparse(triu(A))))
            info = CUSPARSE.sv_analysis('N', 'S', 'U', d_A, 'O')
            d_B  = CUSPARSE.ic0('N', 'S', d_A, info, 'O')
            h_A  = to_host(d_B)
            Ac   = sparse(full(cholfact(A)))
            h_A  = transpose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
    end
    @testset for elty in [Complex64,Complex128]
        @testset "csr" begin
            A    = rand(elty,m,m)
            A    = A + ctranspose(A)
            A   += m * eye(elty,m)
            d_A  = Hermitian(CudaSparseMatrixCSR(sparse(triu(A))))
            info = CUSPARSE.sv_analysis('N', 'H', 'U', d_A, 'O')
            d_B  = CUSPARSE.ic0('N', 'H', d_A, info, 'O')
            h_A  = to_host(d_B)
            Ac   = sparse(full(cholfact(A)))
            h_A  = ctranspose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
        @testset "csc" begin
            A    = rand(elty,m,m)
            A    = A + ctranspose(A)
            A   += m * eye(elty,m)
            d_A  = Hermitian(CudaSparseMatrixCSC(sparse(triu(A))))
            info = CUSPARSE.sv_analysis('N', 'H', 'U', d_A, 'O')
            d_B  = CUSPARSE.ic0('N', 'H', d_A, info, 'O')
            h_A  = to_host(d_B)
            Ac   = sparse(full(cholfact(A)))
            h_A  = ctranspose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
    end
end
@testset "ic2" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        @testset "csr" begin
            A   = rand(elty, m, m)
            A  += ctranspose(A)
            A  += m * eye(elty, m)
            d_A = CudaSparseMatrixCSR(sparse(tril(A)))
            d_B = CUSPARSE.ic02(d_A, 'O')
            h_A = to_host(d_B)
            Ac  = sparse(full(cholfact(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
            A   = rand(elty,m,n)
            d_A = CudaSparseMatrixCSR(sparse(tril(A)))
            @test_throws DimensionMismatch CUSPARSE.ic02(d_A, 'O')
        end
        @testset "csc" begin
            A   = rand(elty, m, m)
            A  += ctranspose(A)
            A  += m * eye(elty, m)
            d_A = CudaSparseMatrixCSC(sparse(tril(A)))
            d_B = CUSPARSE.ic02(d_A, 'O')
            h_A = to_host(d_B)
            Ac  = sparse(full(cholfact(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
            A   = rand(elty,m,n)
            d_A = CudaSparseMatrixCSC(sparse(tril(A)))
            @test_throws DimensionMismatch CUSPARSE.ic02(d_A, 'O')
        end
    end
end
