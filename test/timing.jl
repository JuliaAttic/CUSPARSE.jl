using CUSPARSE
using CUDArt
using Base.Test

ms = [256,512,603,1024,1505,2048]
ns = [512,1024,705,2048,1701,4096]
ks = [128,256,256,512,512,2048]
blockdims = [32,64,256,256,512,512,1024]
types = [Float32, Float64, Complex64, Complex128]
p = 0.2

# matrix * vector

function mv(elty,m,n,bD)
    A = convert(SparseMatrixCSC{elty,Int},sprand(m,n,p))
    x = rand(elty,n)
    y = rand(elty,m)
    alpha = rand(elty)
    beta  = rand(elty)
    tic()
    for i in 1:20
        y = alpha * A * x + beta * y
    end
    ti = toq()
    print("Time for ")
    print_with_color(:blue,"CPU m*v ")
    print("with dims ",m," ",n," and type ",elty,": ",ti,"\n")
    tic()
    d_A = CudaSparseMatrixCSR(A)
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    for i in 1:20
        d_y = CUSPARSE.csrmv!('N',alpha,d_A,d_x,beta,d_y,'O')
    end
    h_y = to_host(d_y)
    ti = toq()
    print("Time for ")
    print_with_color(:purple,"CSR m*v ")
    print("with dims ",m," ",n," and type ",elty,": ",ti,"\n")
    tic()
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,bD))
    d_x = CudaArray(x)
    d_y = CudaArray(y)
    for i in 1:20
        d_y = CUSPARSE.bsrmv!('N',alpha,d_A,d_x,beta,d_y,'O')
    end
    h_y = to_host(d_y)
    ti = toq()
    println("Time for BSR m*v with dims ",m," ",n," and type ",elty,": ",ti)
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
    println("Time for HYB m*v with dims ",m," ",n," and type ",elty,": ",ti)
end

for (m,n,bD) in zip(ms,ns,blockdims)
    for ty in types
        mv(ty,m,n,bD)
    end
end

# matrix/vector solve

function sv(elty,m,bD)
    A = triu(sprand(m,m,p))
    A += speye(m)
    A = convert(SparseMatrixCSC{elty,Int},A)
    X = rand(elty,m)
    alpha = rand(elty)
    tic()
    for i in 1:20
        X = rand(elty,m)
        y = full(A)\(alpha*X)
    end
    ti = toq()
    print("Time for ")
    print_with_color(:blue,"CPU sv ")
    print("with dim ",m," and type ",elty,": ",ti,"\n")
    tic()
    d_A = CudaSparseMatrixCSR(A)
    d_X = CudaArray(X)
    for i in 1:20
        d_X = CUSPARSE.csrsv2!('N',alpha,d_A,d_X,'O')
    end
    h_X = to_host(d_X)
    ti = toq()
    print("Time for ")
    print_with_color(:purple,"CSR sv ")
    print("with dim ",m," and type ",elty,": ",ti,"\n")
    tic()
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,bD))
    d_X = CudaArray(X)
    for i in 1:20
        d_X = CUSPARSE.bsrsv2!('N',alpha,d_A,d_X,'O')
    end
    h_X = to_host(d_X)
    ti = toq()
    println("Time for BSR sv with dim ",m," and type ",elty,": ",ti)
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
    println("Time for HYB sv with dim ",m," and type ",elty,": ",ti)
end

for (m,bD) in zip(ms,blockdims)
    for ty in types
        sv(ty,m,bD)
    end
end

# matrix * matrix

function mm(elty,m,k,n,bD)
    A = convert(SparseMatrixCSC{elty,Int},sprand(m,k,p))
    B = rand(elty,k,n)
    C = rand(elty,m,n)
    alpha = rand(elty)
    beta  = rand(elty)
    tic()
    for i in 1:20
        C = alpha * A * B + beta * C
    end
    ti = toq()
    print("Time for ")
    print_with_color(:blue,"CPU m*m ")
    print("with dims ",m," ",n," and type ",elty,": ",ti,"\n")
    tic()
    d_A = CudaSparseMatrixCSR(A)
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    for i in 1:20
        d_C = CUSPARSE.csrmm2!('N','N',alpha,d_A,d_B,beta,d_C,'O')
    end
    h_C = to_host(d_C)
    ti = toq()
    print("Time for ")
    print_with_color(:purple,"CSR m*m ")
    print("with dims ",m," ",n," and type ",elty,": ",ti,"\n")
    tic()
    d_A = CudaSparseMatrixCSR(A)
    d_A = CUSPARSE.switch2bsr(d_A,convert(Cint,bD))
    d_B = CudaArray(B)
    d_C = CudaArray(C)
    for i in 1:20
        d_C = CUSPARSE.bsrmm!('N','N',alpha,d_A,d_B,beta,d_C,'O')
    end
    h_C = to_host(d_C)
    ti = toq()
    println("Time for BSR m*m with dims ",m," ",n," and type ",elty,": ",ti)
    tic()
end

for (m,n,k,bD) in zip(ms,ns,ks,blockdims)
    for ty in types
        mm(ty,m,k,n,bD)
    end
end

