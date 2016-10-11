using CUSPARSE
using CUDArt
using Base.Test

m = 25
n = 35
k = 10
blockdim = 5

#############
# test_gtsv #
#############

function test_gtsv!(elty)
    dl = rand(elty,m-1)
    du = rand(elty,m-1)
    d = rand(elty,m)
    B = rand(elty,m,n)
    d_dl = CudaArray(vcat([0],dl))
    d_du = CudaArray(vcat(du,[0]))
    d_d = CudaArray(d)
    d_B = CudaArray(B)
    d_B = CUSPARSE.gtsv!(d_dl,d_d,d_du,d_B)
    C = diagm(d,0) + diagm(du,1) + diagm(dl,-1)
    h_B = to_host(d_B)
    @test h_B ≈ C\B
end

function test_gtsv(elty)
    dl = rand(elty,m-1)
    du = rand(elty,m-1)
    d = rand(elty,m)
    B = rand(elty,m,n)
    d_dl = CudaArray(vcat([0],dl))
    d_du = CudaArray(vcat(du,[0]))
    d_d = CudaArray(d)
    d_B = CudaArray(B)
    d_C = CUSPARSE.gtsv(d_dl,d_d,d_du,d_B)
    C = diagm(d,0) + diagm(du,1) + diagm(dl,-1)
    h_C = to_host(d_C)
    @test h_C ≈ C\B
end

#####################
# test_gtsv_nopivot #
#####################

function test_gtsv_nopivot!(elty)
    dl = rand(elty,m-1)
    du = rand(elty,m-1)
    d = rand(elty,m)
    B = rand(elty,m,n)
    d_dl = CudaArray(vcat([0],dl))
    d_du = CudaArray(vcat(du,[0]))
    d_d = CudaArray(d)
    d_B = CudaArray(B)
    d_B = CUSPARSE.gtsv_nopivot!(d_dl,d_d,d_du,d_B)
    C = diagm(d,0) + diagm(du,1) + diagm(dl,-1)
    h_B = to_host(d_B)
    @test h_B ≈ C\B
end

function test_gtsv_nopivot(elty)
    dl = rand(elty,m-1)
    du = rand(elty,m-1)
    d = rand(elty,m)
    B = rand(elty,m,n)
    d_dl = CudaArray(vcat([0],dl))
    d_du = CudaArray(vcat(du,[0]))
    d_d = CudaArray(d)
    d_B = CudaArray(B)
    d_C = CUSPARSE.gtsv_nopivot(d_dl,d_d,d_du,d_B)
    C = diagm(d,0) + diagm(du,1) + diagm(dl,-1)
    h_C = to_host(d_C)
    @test h_C ≈ C\B
end

#########################
# test_gtsvStridedBatch #
#########################

function test_gtsvStridedBatch!(elty)
    dla = rand(elty,m-1)
    dua = rand(elty,m-1)
    da = rand(elty,m)
    dlb = rand(elty,m-1)
    dub = rand(elty,m-1)
    db = rand(elty,m)
    xa = rand(elty,m)
    xb = rand(elty,m)
    d_dl = CudaArray(vcat([0],dla,[0],dlb))
    d_du = CudaArray(vcat(dua,[0],dub,[0]))
    d_d = CudaArray(vcat(da,db))
    d_x = CudaArray(vcat(xa,xb))
    d_x = CUSPARSE.gtsvStridedBatch!(d_dl,d_d,d_du,d_x,2,m)
    Ca = diagm(da,0) + diagm(dua,1) + diagm(dla,-1)
    Cb = diagm(db,0) + diagm(dub,1) + diagm(dlb,-1)
    h_x = to_host(d_x)
    @test h_x[1:m] ≈ Ca\xa
    @test h_x[m+1:2*m] ≈ Cb\xb
end

function test_gtsvStridedBatch(elty)
    dla = rand(elty,m-1)
    dua = rand(elty,m-1)
    da = rand(elty,m)
    dlb = rand(elty,m-1)
    dub = rand(elty,m-1)
    db = rand(elty,m)
    xa = rand(elty,m)
    xb = rand(elty,m)
    d_dl = CudaArray(vcat([0],dla,[0],dlb))
    d_du = CudaArray(vcat(dua,[0],dub,[0]))
    d_d = CudaArray(vcat(da,db))
    d_x = CudaArray(vcat(xa,xb))
    d_y = CUSPARSE.gtsvStridedBatch(d_dl,d_d,d_du,d_x,2,m)
    Ca = diagm(da,0) + diagm(dua,1) + diagm(dla,-1)
    Cb = diagm(db,0) + diagm(dub,1) + diagm(dlb,-1)
    h_y = to_host(d_y)
    @test h_y[1:m] ≈ Ca\xa
    @test h_y[m+1:2*m] ≈ Cb\xb
end

types = [Float32,Float64,Complex64,Complex128]
for elty in types
    tic()
    test_gtsv!(elty)
    println("gtsv! took ", toq(), " for ", elty)
    tic()
    test_gtsv(elty)
    println("gtsv took ", toq(), " for ", elty)
    tic()
    test_gtsv_nopivot!(elty)
    println("gtsv_nopivot! took ", toq(), " for ", elty)
    tic()
    test_gtsv_nopivot(elty)
    println("gtsv_nopivot took ", toq(), " for ", elty)
    tic()
    test_gtsvStridedBatch!(elty)
    println("gtsvStridedBatch! took ", toq(), " for ", elty)
    tic()
    test_gtsvStridedBatch(elty)
    println("gtsvStridedBatch took ", toq(), " for ", elty)
end
