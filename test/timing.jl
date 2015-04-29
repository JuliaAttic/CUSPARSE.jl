using CUSPARSE
using CUDArt
using Base.Test
using Gadfly

ms = [256,512,603,1024,1505,2048]
ns = [512,1024,705,2048,1701,4096]
ks = [128,256,256,512,512,2048]
blockdims = [32,64,256,256,512,512,1024]
types = [Float32, Float64, Complex64, Complex128]
prob = 0.2

# matrix * vector
function cpumv(elty,m,n,arr)
    A = convert(SparseMatrixCSC{elty,Int},sprand(m,n,prob))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta  = rand(elty)
    tic()
    for i in 1:20
        y = alpha * A * x + beta * y
    end
    ti = toq()
    return vcat(arr,[ti])
end

function csrmv(elty,m,n,arr)
    A = convert(SparseMatrixCSC{elty,Int},sprand(m,n,prob))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta  = rand(elty)
    tic()
    d_A = CudaSparseMatrixCSR(A)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    for i in 1:20
        d_y = CUSPARSE.csrmv!('N',alpha,d_A,d_x,beta,d_y,'O')
    end
    h_y = to_host(d_y)
    ti = toq()
    return vcat(arr,[ti])
end

function hybmv(elty,m,n,arr)
    A = convert(SparseMatrixCSC{elty,Int},sprand(m,n,prob))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta  = rand(elty)
    tic()
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2hyb(d_A)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    for i in 1:20
        d_y = CUSPARSE.hybmv!('N',alpha,d_A,d_x,beta,d_y,'O')
    end
    h_y = to_host(d_y)
    ti = toq()
    return vcat(arr,[ti])
end

p = []
for ty in types
    mv = Gadfly.DataFrame()
    mvcputimes = Any[]
    mvcsrtimes = Any[]
    mvhybtimes = Any[]
    xind = Any[]
    for (m,n) in zip(ms,ns)
        mvcputimes = cpumv(ty,m,n,mvcputimes)
        mvcsrtimes = csrmv(ty,m,n,mvcsrtimes)
        mvhybtimes = hybmv(ty,m,n,mvhybtimes)
        xind = vcat(xind,(m,n))
    end
    mv = vcat(mv, Gadfly.DataFrame(x = xind, y = ones(length(ms)), Format=string("CPU")))
    mv = vcat(mv, Gadfly.DataFrame(x = xind, y = mvcputimes./mvcsrtimes, Format=string("CSR")))
    mv = vcat(mv, Gadfly.DataFrame(x = xind, y = mvcputimes./mvhybtimes, Format=string("HYB")))
    p = vcat(p,Gadfly.plot(mv, x="x",y="y",color="Format",Stat.yticks([0:5:40]),Scale.x_discrete,Geom.bar(position=:dodge),Guide.xlabel("Matrix size"),Guide.ylabel("Speedup"),Guide.title(string("Speed comparison for 20 sm * dv multiplications with precision ",ty))))
end

draw(PDF(("mv.pdf"),15inch,15inch),vstack(hstack(p[1],p[2]),hstack(p[3],p[4])))

# matrix/vector solve

function cpusv(elty,m,arr)
    A = triu(sprand(m,m,prob))
    A += speye(m)
    A = convert(SparseMatrixCSC{elty,Cint},A)
    X = rand(elty,m)
    alpha = rand(elty)
    tic()
    for i in 1:20
        X = rand(elty,m)
        y = full(A)\(alpha*X)
    end
    ti = toq()
    return vcat(arr,[ti])
end

function csrsv(elty,m,arr)
    A = triu(sprand(m,m,prob))
    A += speye(m)
    A = convert(SparseMatrixCSC{elty,Cint},A)
    X = rand(elty,m)
    alpha = rand(elty)
    tic()
    d_A = CudaSparseMatrixCSR(A)
    d_X = CudaArray(X)
    for i in 1:20
        d_X = CUSPARSE.csrsv2!('N',alpha,d_A,d_X,'O')
    end
    h_X = to_host(d_X)
    ti = toq()
    return vcat(arr,[ti])
end

function hybsv(elty,m,arr)
    A = triu(sprand(m,m,prob))
    A += speye(m)
    A = convert(SparseMatrixCSC{elty,Cint},A)
    X = rand(elty,m)
    alpha = rand(elty)
    tic()
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2hyb(d_A)
    d_X = CudaArray(X)
    for i in 1:20
        info = CUSPARSE.hybsv_analysis('N','U',d_A,'O')
        d_X = CUSPARSE.hybsv_solve('N','U',alpha,d_A,d_X,info,'O')
    end
    h_X = to_host(d_X)
    ti = toq()
    return vcat(arr,[ti])
end

p = []
for ty in types
    sv = Gadfly.DataFrame()
    svcputimes = Any[]
    svcsrtimes = Any[]
    svhybtimes = Any[]
    xind = Any[]
    for (m,n) in zip(ms,ns)
        svcputimes = cpusv(ty,m,svcputimes)
        svcsrtimes = csrsv(ty,m,svcsrtimes)
        svhybtimes = hybsv(ty,m,svhybtimes)
        xind = vcat(xind,(m,n))
    end
    sv = vcat(sv, Gadfly.DataFrame(x = xind, y = ones(length(ms)), Format=string("CPU")))
    sv = vcat(sv, Gadfly.DataFrame(x = xind, y = svcputimes./svcsrtimes, Format=string("CSR")))
    sv = vcat(sv, Gadfly.DataFrame(x = xind, y = svcputimes./svhybtimes, Format=string("HYB")))
    p = vcat(p,Gadfly.plot(sv, x="x",y="y",color="Format",Stat.yticks([0:5:40]),Scale.x_discrete,Geom.bar(position=:dodge),Guide.xlabel("Matrix size"),Guide.ylabel("Speedup"),Guide.title(string("Speed comparison for 20 sm \ dv solutions with precision ",ty))))
end

draw(PDF(("sv.pdf"),15inch,15inch),vstack(hstack(p[1],p[2]),hstack(p[3],p[4])))

# matrix * matrix

function cpumm(elty,m,k,n,arr)
    A = convert(SparseMatrixCSC{elty,Int},sprand(m,k,prob))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta  = rand(elty)
    tic()
    for i in 1:20
        C = alpha * A * B + beta * C
    end
    ti = toq()
    return vcat(arr,[ti])
end

function csrmm(elty,m,k,n,arr)
    A = convert(SparseMatrixCSC{elty,Int},sprand(m,k,prob))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta  = rand(elty)
    tic()
    d_A = CudaSparseMatrixCSR(A)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    for i in 1:20
        d_C = CUSPARSE.csrmm2!('N','N',alpha,d_A,d_B,beta,d_C,'O')
    end
    h_C = to_host(d_C)
    ti = toq()
    return vcat(arr,[ti])
end

p = []
for ty in types
    mm = Gadfly.DataFrame()
    mmcputimes = Any[]
    mmcsrtimes = Any[]
    xind = Any[]
    for (m,k,n) in zip(ms,ks,ns)
        mmcputimes = cpumm(ty,m,k,n,mmcputimes)
        mmcsrtimes = csrmm(ty,m,k,n,mmcsrtimes)
        xind = vcat(xind,(m,k,n))
    end
    mm = vcat(mm, Gadfly.DataFrame(x = xind, y = ones(length(ms)), Format=string("CPU")))
    mm = vcat(mm, Gadfly.DataFrame(x = xind, y = mmcputimes./mmcsrtimes, Format=string("CSR")))
    p = vcat(p,Gadfly.plot(mm, x="x",y="y",color="Format",Stat.yticks([0:5:50]),Scale.x_discrete,Geom.bar(position=:dodge),Guide.xlabel("Matrix sizes"),Guide.ylabel("Speedup"),Guide.title(string("Speed comparison for 20 sm * sm multiplications with precision ",ty))))
end

draw(PDF(("mm.pdf"),15inch,15inch),vstack(hstack(p[1],p[2]),hstack(p[3],p[4])))
