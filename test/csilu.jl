using CUSPARSE
using CUDAdrv
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "ilu0" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        A = rand(elty,m,m)
        @testset "csr" begin
            d_A = CudaSparseMatrixCSR(sparse(A))
            info = CUSPARSE.sv_analysis('N','G','U',d_A,'O')
            d_B = CUSPARSE.ilu0('N',d_A,info,'O')
            h_B = to_host(d_B)
            Alu = lufact(full(A),Val{false})
            Ac = sparse(Alu[:L]*Alu[:U])
            h_A = ctranspose(h_B) * h_B
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
        @testset "csc" begin
            d_A = CudaSparseMatrixCSC(sparse(A))
            info = CUSPARSE.sv_analysis('N','G','U',d_A,'O')
            d_B = CUSPARSE.ilu0('N',d_A,info,'O')
            h_B = to_host(d_B)
            Alu = lufact(full(A),Val{false})
            Ac = sparse(Alu[:L]*Alu[:U])
            h_A = ctranspose(h_B) * h_B
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
    end
end

@testset "ilu02" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        @testset "csr" begin
            A = rand(elty,m,m)
            A += transpose(A)
            A += m * eye(elty,m)
            d_A = CudaSparseMatrixCSR(sparse(A))
            d_B = CUSPARSE.ilu02(d_A,'O')
            h_A = to_host(d_B)
            Alu = lufact(full(A),Val{false})
            Ac = sparse(Alu[:L]*Alu[:U])
            h_A = ctranspose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
        @testset "csc" begin
            A = rand(elty,m,m)
            A += transpose(A)
            A += m * eye(elty,m)
            d_A = CudaSparseMatrixCSC(sparse(A))
            d_B = CUSPARSE.ilu02(d_A,'O')
            h_A = to_host(d_B)
            Alu = lufact(full(A),Val{false})
            Ac = sparse(Alu[:L]*Alu[:U])
            h_A = ctranspose(h_A) * h_A
            @test h_A.rowval ≈ Ac.rowval
            @test reduce(&, isfinite.(h_A.nzval))
        end
    end
end
