using CUSPARSE
using CuArrays
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

@testset "gtsv" begin
    @testset for elty in [Float32,Float64,Complex64,Complex128]
        @testset "gtsv!" begin
            dl = rand(elty,m-1)
            du = rand(elty,m-1)
            d = rand(elty,m)
            B = rand(elty,m,n)
            d_dl = CuArray(vcat([0],dl))
            d_du = CuArray(vcat(du,[0]))
            d_d = CuArray(d)
            d_B = CuArray(B)
            d_B = CUSPARSE.gtsv!(d_dl,d_d,d_du,d_B)
            C = diagm(d,0) + diagm(du,1) + diagm(dl,-1)
            h_B = collect(d_B)
            @test h_B ≈ C\B
        end

        @testset "gtsv" begin
            dl = rand(elty,m-1)
            du = rand(elty,m-1)
            d = rand(elty,m)
            B = rand(elty,m,n)
            d_dl = CuArray(vcat([0],dl))
            d_du = CuArray(vcat(du,[0]))
            d_d = CuArray(d)
            d_B = CuArray(B)
            d_C = CUSPARSE.gtsv(d_dl,d_d,d_du,d_B)
            C = diagm(d,0) + diagm(du,1) + diagm(dl,-1)
            h_C = collect(d_C)
            @test h_C ≈ C\B
        end

        @testset "gtsv_nopivot!" begin
            dl = rand(elty,m-1)
            du = rand(elty,m-1)
            d = rand(elty,m)
            B = rand(elty,m,n)
            d_dl = CuArray(vcat([0],dl))
            d_du = CuArray(vcat(du,[0]))
            d_d = CuArray(d)
            d_B = CuArray(B)
            d_B = CUSPARSE.gtsv_nopivot!(d_dl,d_d,d_du,d_B)
            C = diagm(d,0) + diagm(du,1) + diagm(dl,-1)
            h_B = collect(d_B)
            @test h_B ≈ C\B
        end

        @testset "gtsv_nopivot" begin
            dl = rand(elty,m-1)
            du = rand(elty,m-1)
            d = rand(elty,m)
            B = rand(elty,m,n)
            d_dl = CuArray(vcat([0],dl))
            d_du = CuArray(vcat(du,[0]))
            d_d = CuArray(d)
            d_B = CuArray(B)
            d_C = CUSPARSE.gtsv_nopivot(d_dl,d_d,d_du,d_B)
            C = diagm(d,0) + diagm(du,1) + diagm(dl,-1)
            h_C = collect(d_C)
            @test h_C ≈ C\B
        end

        @testset "gtsvStridedBatch!" begin
            dla = rand(elty,m-1)
            dua = rand(elty,m-1)
            da = rand(elty,m)
            dlb = rand(elty,m-1)
            dub = rand(elty,m-1)
            db = rand(elty,m)
            xa = rand(elty,m)
            xb = rand(elty,m)
            d_dl = CuArray(vcat([0],dla,[0],dlb))
            d_du = CuArray(vcat(dua,[0],dub,[0]))
            d_d = CuArray(vcat(da,db))
            d_x = CuArray(vcat(xa,xb))
            d_x = CUSPARSE.gtsvStridedBatch!(d_dl,d_d,d_du,d_x,2,m)
            Ca = diagm(da,0) + diagm(dua,1) + diagm(dla,-1)
            Cb = diagm(db,0) + diagm(dub,1) + diagm(dlb,-1)
            h_x = collect(d_x)
            @test h_x[1:m] ≈ Ca\xa
            @test h_x[m+1:2*m] ≈ Cb\xb
        end

        @testset "gtsvStridedBatch" begin
            dla = rand(elty,m-1)
            dua = rand(elty,m-1)
            da = rand(elty,m)
            dlb = rand(elty,m-1)
            dub = rand(elty,m-1)
            db = rand(elty,m)
            xa = rand(elty,m)
            xb = rand(elty,m)
            d_dl = CuArray(vcat([0],dla,[0],dlb))
            d_du = CuArray(vcat(dua,[0],dub,[0]))
            d_d = CuArray(vcat(da,db))
            d_x = CuArray(vcat(xa,xb))
            d_y = CUSPARSE.gtsvStridedBatch(d_dl,d_d,d_du,d_x,2,m)
            Ca = diagm(da,0) + diagm(dua,1) + diagm(dla,-1)
            Cb = diagm(db,0) + diagm(dub,1) + diagm(dlb,-1)
            h_y = collect(d_y)
            @test h_y[1:m] ≈ Ca\xa
            @test h_y[m+1:2*m] ≈ Cb\xb
        end
    end
end
