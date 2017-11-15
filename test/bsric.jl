using CUSPARSE
using CuArrays
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "bsric02" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        A = rand(elty, m, m)
        A += ctranspose(A)
        A += m * eye(elty, m)

        @testset "bsric02!" begin
            d_A = CudaSparseMatrixCSR(sparse(tril(A)))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_A = CUSPARSE.ic02!(d_A,'O')
            h_A = to_host(CUSPARSE.switch2csr(d_A))
            Ac = sparse(full(cholfact(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
            d_A = CudaSparseMatrixCSR(sparse(tril(rand(elty,m,n))))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            @test_throws DimensionMismatch CUSPARSE.ic02!(d_A,'O')
        end

        @testset "bsric02" begin
            d_A = CudaSparseMatrixCSR(sparse(tril(A)))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_B = CUSPARSE.ic02(d_A,'O')
            h_A = to_host(CUSPARSE.switch2csr(d_B))
            Ac = sparse(full(cholfact(Hermitian(A))))
            h_A = transpose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
    end
end
