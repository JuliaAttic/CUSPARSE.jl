using CUSPARSE
using CUDAdrv
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "bsrilu02" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        A = rand(elty,m,m)
        A += transpose(A)
        A += m * eye(elty,m)
        @testset "bsrilu02!" begin
            d_A = CudaSparseMatrixCSR(sparse(A))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_A = CUSPARSE.ilu02!(d_A,'O')
            h_A = to_host(CUSPARSE.switch2csr(d_A))
            Alu = lufact(full(A),Val{false})
            Ac = sparse(Alu[:L]*Alu[:U])
            h_A = ctranspose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
            d_A = CudaSparseMatrixCSR(sparse(rand(elty,m,n)))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            @test_throws DimensionMismatch CUSPARSE.ilu02!(d_A,'O')
        end

        @testset "bsrilu02" begin
            d_A = CudaSparseMatrixCSR(sparse(A))
            d_A = CUSPARSE.switch2bsr(d_A, convert(Cint,5))
            d_B = CUSPARSE.ilu02(d_A,'O')
            h_A = to_host(CUSPARSE.switch2csr(d_B))
            Alu = lufact(full(A),Val{false})
            Ac = sparse(Alu[:L]*Alu[:U])
            h_A = ctranspose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
    end
end
