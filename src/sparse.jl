#utilities

import Base.LinAlg: HermOrSym, AbstractTriangular, *, +, -, \, A_mul_Bt, At_mul_B, At_mul_Bt, Ac_mul_B, At_ldiv_B, Ac_ldiv_B

# convert SparseChar {N,T,C} to cusparseOperation_t
function cusparseop(trans::SparseChar)
    if trans == 'N'
        return CUSPARSE_OPERATION_NON_TRANSPOSE
    end
    if trans == 'T'
        return CUSPARSE_OPERATION_TRANSPOSE
    end
    if trans == 'C'
        return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    end
    throw(ArgumentError("unknown cusparse operation."))
end

# convert SparseChar {G,S,H,T} to cusparseMatrixType_t
function cusparsetype(mattype::SparseChar)
    if mattype == 'G'
        return CUSPARSE_MATRIX_TYPE_GENERAL
    end
    if mattype == 'T'
        return CUSPARSE_MATRIX_TYPE_TRIANGULAR
    end
    if mattype == 'S'
        return CUSPARSE_MATRIX_TYPE_SYMMETRIC
    end
    if mattype == 'H'
        return CUSPARSE_MATRIX_TYPE_HERMITIAN
    end
    throw(ArgumentError("unknown cusparse matrix type."))
end

# convert SparseChar {U,L} to cusparseFillMode_t
function cusparsefill(uplo::SparseChar)
    if uplo == 'U'
        return CUSPARSE_FILL_MODE_UPPER
    end
    if uplo == 'L'
        return CUSPARSE_FILL_MODE_LOWER
    end
    throw(ArgumentError("unknown cusparse fill mode"))
end

# convert SparseChar {U,N} to cusparseDiagType_t
function cusparsediag(diag::SparseChar)
    if diag == 'U'
        return CUSPARSE_DIAG_TYPE_UNIT
    end
    if diag == 'N'
        return CUSPARSE_DIAG_TYPE_NON_UNIT
    end
    throw(ArgumentError("unknown cusparse diag mode"))
end

# convert SparseChar {Z,O} to cusparseIndexBase_t
function cusparseindex(index::SparseChar)
    if index == 'Z'
        return CUSPARSE_INDEX_BASE_ZERO
    end
    if index == 'O'
        return CUSPARSE_INDEX_BASE_ONE
    end
    throw(ArgumentError("unknown cusparse index base"))
end

# convert SparseChar {R,C} to cusparseDirection_t
function cusparsedir(dir::SparseChar)
    if dir == 'R'
        return CUSPARSE_DIRECTION_ROW
    end
    if dir == 'C'
        return CUSPARSE_DIRECTION_COL
    end
    throw(ArgumentError("unknown cusparse direction"))
end

function chkmvdims( X, n, Y, m)
    if length(X) != n
        throw(DimensionMismatch("X must have length $n, but has length $(length(X))"))
    elseif length(Y) != m
        throw(DimensionMismatch("Y must have length $m, but has length $(length(Y))"))
    end
end

function chkmmdims( B, C, k, l, m, n )
    if size(B) != (k,l)
        throw(DimensionMismatch("B has dimensions $(size(B)) but needs ($k,$l)"))
    elseif size(C) != (m,n)
        throw(DimensionMismatch("C has dimensions $(size(C)) but needs ($m,$n)"))
    end
end

function getDescr( A::CudaSparseMatrix, index::SparseChar )
    cuind = cusparseindex(index)
    typ   = CUSPARSE_MATRIX_TYPE_GENERAL
    fill  = CUSPARSE_FILL_MODE_LOWER
    if ishermitian(A)
        typ  = CUSPARSE_MATRIX_TYPE_HERMITIAN
        fill = cusparsefill(A.uplo)
    elseif issym(A)
        typ  = CUSPARSE_MATRIX_TYPE_SYMMETRIC
        fill = cusparsefill(A.uplo)
    end
    cudesc = cusparseMatDescr_t(typ, fill,CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
end

function getDescr( A::Symmetric, index::SparseChar )
    cuind = cusparseindex(index)
    typ  = CUSPARSE_MATRIX_TYPE_SYMMETRIC
    fill = cusparsefill(A.uplo)
    cudesc = cusparseMatDescr_t(typ, fill,CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
end

function getDescr( A::Hermitian, index::SparseChar )
    cuind = cusparseindex(index)
    typ  = CUSPARSE_MATRIX_TYPE_HERMITIAN
    fill = cusparsefill(A.uplo)
    cudesc = cusparseMatDescr_t(typ, fill,CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
end

# type conversion
for (fname,elty) in ((:cusparseScsr2csc, :Float32),
                     (:cusparseDcsr2csc, :Float64),
                     (:cusparseCcsr2csc, :Complex64),
                     (:cusparseZcsr2csc, :Complex128))
    @eval begin
        function switch2csc(csr::CudaSparseMatrixCSR{$elty},inda::SparseChar='O')
            cuind = cusparseindex(inda)
            m,n = csr.dims
            colPtr = CudaArray(zeros(Cint,n+1))
            rowVal = CudaArray(zeros(Cint,csr.nnz))
            nzVal = CudaArray(zeros($elty,csr.nnz))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, Cint, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, cusparseAction_t, cusparseIndexBase_t),
                               cusparsehandle[1], m, n, csr.nnz, csr.nzVal,
                               csr.rowPtr, csr.colVal, nzVal, rowVal,
                               colPtr, CUSPARSE_ACTION_NUMERIC, cuind))
            csc = CudaSparseMatrixCSC(colPtr,rowVal,nzVal,csr.nnz,csr.dims)
            csc
        end
        function switch2csr(csc::CudaSparseMatrixCSC{$elty},inda::SparseChar='O')
            cuind = cusparseindex(inda)
            m,n = csc.dims
            rowPtr = CudaArray(zeros(Cint,m+1))
            colVal = CudaArray(zeros(Cint,csc.nnz))
            nzVal = CudaArray(zeros($elty,csc.nnz))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, Cint, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, cusparseAction_t, cusparseIndexBase_t),
                               cusparsehandle[1], n, m, csc.nnz, csc.nzVal,
                               csc.colPtr, csc.rowVal, nzVal, colVal,
                               rowPtr, CUSPARSE_ACTION_NUMERIC, cuind))
            csr = CudaSparseMatrixCSR(rowPtr,colVal,nzVal,csc.nnz,csc.dims)
            csr
        end
    end
end

for (fname,elty) in ((:cusparseScsr2bsr, :Float32),
                     (:cusparseDcsr2bsr, :Float64),
                     (:cusparseCcsr2bsr, :Complex64),
                     (:cusparseZcsr2bsr, :Complex128))
    @eval begin
        function switch2bsr(csr::CudaSparseMatrixCSR{$elty},
                            blockDim::Cint,
                            dir::SparseChar='R',
                            inda::SparseChar='O',
                            indc::SparseChar='O')
            cudir = cusparsedir(dir)
            cuinda = cusparseindex(inda)
            cuindc = cusparseindex(indc)
            m,n = csr.dims
            nnz = Array(Cint,1)
            mb = div((m + blockDim - 1),blockDim)
            nb = div((n + blockDim - 1),blockDim)
            bsrRowPtr = CudaArray(zeros(Cint,mb + 1))
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            statuscheck(ccall((:cusparseXcsr2bsrNnz,libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{Cint},
                               Ptr{Cint}, Cint, Ptr{cusparseMatDescr_t},
                               Ptr{Cint}, Ptr{Cint}),
                              cusparsehandle[1], cudir, m, n, &cudesca, csr.rowPtr,
                              csr.colVal, blockDim, &cudescc, bsrRowPtr, nnz))
            bsrNzVal = CudaArray(zeros($elty, nnz[1] * blockDim * blockDim ))
            bsrColInd = CudaArray(zeros(Cint, nnz[1]))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}), cusparsehandle[1], cudir, m, n,
                               &cudesca, csr.nzVal, csr.rowPtr, csr.colVal,
                               blockDim, &cudescc, bsrNzVal, bsrRowPtr,
                               bsrColInd))
            CudaSparseMatrixBSR{$elty}(bsrRowPtr, bsrColInd, bsrNzVal, csr.dims, blockDim, dir, nnz[1], csr.dev)
        end
        function switch2bsr(csc::CudaSparseMatrixCSC{$elty},
                            blockDim::Cint,
                            dir::SparseChar='R',
                            inda::SparseChar='O',
                            indc::SparseChar='O')
                switch2bsr(switch2csr(csc),blockDim,dir,inda,indc)
        end
    end
end

for (fname,elty) in ((:cusparseSbsr2csr, :Float32),
                     (:cusparseDbsr2csr, :Float64),
                     (:cusparseCbsr2csr, :Complex64),
                     (:cusparseZbsr2csr, :Complex128))
    @eval begin
        function switch2csr(bsr::CudaSparseMatrixBSR{$elty},
                            inda::SparseChar='O',
                            indc::SparseChar='O')
            cudir = cusparsedir(bsr.dir)
            cuinda = cusparseindex(inda)
            cuindc = cusparseindex(indc)
            m,n = bsr.dims
            mb = div(m,bsr.blockDim)
            nb = div(n,bsr.blockDim)
            nnz = bsr.nnz * bsr.blockDim * bsr.blockDim
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            csrRowPtr = CudaArray(zeros(Cint, m + 1))
            csrColInd = CudaArray(zeros(Cint, nnz))
            csrNzVal  = CudaArray(zeros($elty, nnz))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}), cusparsehandle[1], cudir, mb, nb,
                               &cudesca, bsr.nzVal, bsr.rowPtr, bsr.colVal,
                               bsr.blockDim, &cudescc, csrNzVal, csrRowPtr,
                               csrColInd))
            CudaSparseMatrixCSR(csrRowPtr, csrColInd, csrNzVal, convert(Cint,nnz), bsr.dims)
        end
        function switch2csc(bsr::CudaSparseMatrixBSR{$elty},
                            inda::SparseChar='O',
                            indc::SparseChar='O')
            switch2csc(switch2csr(bsr,inda,indc))
        end
    end
end

for (cname,rname,elty) in ((:cusparseScsc2dense, :cusparseScsr2dense, :Float32),
                           (:cusparseDcsc2dense, :cusparseDcsr2dense, :Float64),
                           (:cusparseCcsc2dense, :cusparseCcsr2dense, :Complex64),
                           (:cusparseZcsc2dense, :cusparseZcsr2dense, :Complex128))
    @eval begin
        function full(csr::CudaSparseMatrixCSR{$elty},ind::SparseChar='O')
            cuind = cusparseindex(ind)
            m,n = csr.dims
            denseA = CudaArray(zeros($elty,m,n))
            lda = max(1,stride(denseA,2))
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            statuscheck(ccall(($(string(rname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Cint),
                               cusparsehandle[1], m, n, &cudesc, csr.nzVal,
                               csr.rowPtr, csr.colVal, denseA, lda))
            denseA
        end
        function full(csc::CudaSparseMatrixCSC{$elty},ind::SparseChar='O')
            cuind = cusparseindex(ind)
            m,n = csc.dims
            denseA = CudaArray(zeros($elty,m,n))
            lda = max(1,stride(denseA,2))
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            statuscheck(ccall(($(string(cname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Cint),
                               cusparsehandle[1], m, n, &cudesc, csc.nzVal,
                               csc.rowVal, csc.colPtr, denseA, lda))
            denseA
        end
        function full(hyb::CudaSparseMatrixHYB{$elty},ind::SparseChar='O')
            full(switch2csr(hyb,ind))
        end
        function full(bsr::CudaSparseMatrixBSR{$elty},ind::SparseChar='O')
            full(switch2csr(bsr,ind))
        end
    end
end

for (nname,cname,rname,hname,elty) in ((:cusparseSnnz, :cusparseSdense2csc, :cusparseSdense2csr, :cusparseSdense2hyb, :Float32),
                                       (:cusparseDnnz, :cusparseDdense2csc, :cusparseDdense2csr, :cusparseDdense2hyb, :Float64),
                                       (:cusparseCnnz, :cusparseCdense2csc, :cusparseCdense2csr, :cusparseCdense2hyb, :Complex64),
                                       (:cusparseZnnz, :cusparseZdense2csc, :cusparseZdense2csr, :cusparseZdense2hyb, :Complex128))
    @eval begin
        function sparse(A::CudaMatrix{$elty},fmt::SparseChar='R',ind::SparseChar='O')
            cuind = cusparseindex(ind)
            cudir = cusparsedir('R')
            if( fmt == 'C' )
                cudir = cusparsedir(fmt)
            end
            m,n = size(A)
            lda = max(1,stride(A,2))
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            nnzRowCol = CudaArray(zeros(Cint, fmt == 'R' ? m : n))
            nnzTotal = Array(Cint,1)
            statuscheck(ccall(($(string(nname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Cint, Ptr{Cint}, Ptr{Cint}), cusparsehandle[1],
                               cudir, m, n, &cudesc, A, lda, nnzRowCol,
                               nnzTotal))
            nzVal = CudaArray(zeros($elty,nnzTotal[1]))
            if(fmt == 'R')
                rowPtr = CudaArray(zeros(Cint,m+1))
                colInd = CudaArray(zeros(Cint,nnzTotal[1]))
                statuscheck(ccall(($(string(rname)),libcusparse), cusparseStatus_t,
                                  (cusparseHandle_t, Cint, Cint,
                                   Ptr{cusparseMatDescr_t}, Ptr{$elty},
                                   Cint, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                                   Ptr{Cint}), cusparsehandle[1], m, n, &cudesc, A,
                                   lda, nnzRowCol, nzVal, rowPtr, colInd))
                return CudaSparseMatrixCSR(rowPtr,colInd,nzVal,nnzTotal[1],size(A))
            end
            if(fmt == 'C')
                colPtr = CudaArray(zeros(Cint,n+1))
                rowInd = CudaArray(zeros(Cint,nnzTotal[1]))
                statuscheck(ccall(($(string(cname)),libcusparse), cusparseStatus_t,
                                  (cusparseHandle_t, Cint, Cint,
                                   Ptr{cusparseMatDescr_t}, Ptr{$elty},
                                   Cint, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                                   Ptr{Cint}), cusparsehandle[1], m, n, &cudesc, A,
                                   lda, nnzRowCol, nzVal, rowInd, colPtr))
                return CudaSparseMatrixCSC(colPtr,rowInd,nzVal,nnzTotal[1],size(A))
            end
            if(fmt == 'B')
                return switch2bsr(sparse(A,'R',ind),convert(Cint,gcd(m,n)))
            end
            if(fmt == 'H')
                hyb = cusparseHybMat_t[0]
                statuscheck(ccall((:cusparseCreateHybMat,libcusparse), cusparseStatus_t,
                                  (Ptr{cusparseHybMat_t},), hyb))
                statuscheck(ccall(($(string(hname)),libcusparse), cusparseStatus_t,
                                  (cusparseHandle_t, Cint, Cint,
                                   Ptr{cusparseMatDescr_t}, Ptr{$elty},
                                   Cint, Ptr{Cint}, cusparseHybMat_t,
                                   Cint, cusparseHybPartition_t),
                                  cusparsehandle[1], m, n, &cudesc, A, lda, nnzRowCol,
                                  hyb[1], 0, CUSPARSE_HYB_PARTITION_AUTO))
                return CudaSparseMatrixHYB{$elty}(hyb[1],size(A),nnzTotal[1],device())
            end
        end
    end
end

for (rname,cname,elty) in ((:cusparseScsr2hyb, :cusparseScsc2hyb, :Float32),
                           (:cusparseDcsr2hyb, :cusparseDcsc2hyb, :Float64),
                           (:cusparseCcsr2hyb, :cusparseCcsc2hyb, :Complex64),
                           (:cusparseZcsr2hyb, :cusparseZcsc2hyb, :Complex128))
    @eval begin
        function switch2hyb(csr::CudaSparseMatrixCSR{$elty},
                            inda::SparseChar='O')
            cuinda = cusparseindex(inda)
            m,n = csr.dims
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            hyb = cusparseHybMat_t[0]
            statuscheck(ccall((:cusparseCreateHybMat,libcusparse), cusparseStatus_t,
                              (Ptr{cusparseHybMat_t},), hyb))
            statuscheck(ccall(($(string(rname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, cusparseHybMat_t,
                               Cint, cusparseHybPartition_t), cusparsehandle[1],
                               m, n, &cudesca, csr.nzVal, csr.rowPtr, csr.colVal,
                               hyb[1], 0, CUSPARSE_HYB_PARTITION_AUTO))
            CudaSparseMatrixHYB{$elty}(hyb[1], csr.dims, csr.nnz, csr.dev)
        end
        function switch2hyb(csc::CudaSparseMatrixCSC{$elty},
                            inda::SparseChar='O')
            switch2hyb(switch2csr(csc,inda),inda)
        end
    end
end

for (rname,cname,elty) in ((:cusparseShyb2csr, :cusparseShyb2csc, :Float32),
                           (:cusparseDhyb2csr, :cusparseDhyb2csc, :Float64),
                           (:cusparseChyb2csr, :cusparseChyb2csc, :Complex64),
                           (:cusparseZhyb2csr, :cusparseZhyb2csc, :Complex128))
    @eval begin
        function switch2csr(hyb::CudaSparseMatrixHYB{$elty},
                            inda::SparseChar='O')
            cuinda = cusparseindex(inda)
            m,n = hyb.dims
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            csrRowPtr = CudaArray(zeros(Cint, m + 1))
            csrColInd = CudaArray(zeros(Cint, hyb.nnz))
            csrNzVal = CudaArray(zeros($elty, hyb.nnz))
            statuscheck(ccall(($(string(rname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Ptr{cusparseMatDescr_t},
                               cusparseHybMat_t, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}), cusparsehandle[1], &cudesca,
                               hyb.Mat, csrNzVal, csrRowPtr, csrColInd))
            CudaSparseMatrixCSR(csrRowPtr, csrColInd, csrNzVal, hyb.nnz, hyb.dims)
        end
        function switch2csc(hyb::CudaSparseMatrixHYB{$elty},
                            inda::SparseChar='O')
            switch2csc(switch2csr(hyb,inda),inda)
        end
    end
end

# Level 1 CUSPARSE functions

for (fname,elty) in ((:cusparseSaxpyi, :Float32),
                     (:cusparseDaxpyi, :Float64),
                     (:cusparseCaxpyi, :Complex64),
                     (:cusparseZaxpyi, :Complex128))
    @eval begin
        function axpyi!(alpha::$elty,
                        X::CudaSparseVector{$elty},
                        Y::CudaVector{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{$elty},
                               Ptr{Cint}, Ptr{$elty}, cusparseIndexBase_t),
                              cusparsehandle[1], X.nnz, [alpha], X.nzVal, X.iPtr,
                              Y, cuind))
            Y
        end
        function axpyi(alpha::$elty,
                       X::CudaSparseVector{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            axpyi!(alpha,X,copy(Y),index)
        end
        function axpyi(X::CudaSparseVector{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            axpyi!(one($elty),X,copy(Y),index)
        end
    end
end

for (jname,fname,elty) in ((:doti, :cusparseSdoti, :Float32),
                           (:doti, :cusparseDdoti, :Float64),
                           (:doti, :cusparseCdoti, :Complex64),
                           (:doti, :cusparseZdoti, :Complex128),
                           (:dotci, :cusparseCdotci, :Complex64),
                           (:dotci, :cusparseZdotci, :Complex128))
    @eval begin
        function $jname(X::CudaSparseVector{$elty},
                        Y::CudaVector{$elty},
                        index::SparseChar)
            dot = Array($elty,1)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{Cint},
                               Ptr{$elty}, Ptr{$elty}, cusparseIndexBase_t),
                              cusparsehandle[1], X.nnz, X.nzVal, X.iPtr,
                              Y, dot, cuind))
            return dot[1]
        end
    end
end

for (fname,elty) in ((:cusparseSgthr, :Float32),
                     (:cusparseDgthr, :Float64),
                     (:cusparseCgthr, :Complex64),
                     (:cusparseZgthr, :Complex128))
    @eval begin
        function gthr!(X::CudaSparseVector{$elty},
                      Y::CudaVector{$elty},
                      index::SparseChar)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{$elty},
                               Ptr{Cint}, cusparseIndexBase_t), cusparsehandle[1],
                              X.nnz, Y, X.nzVal, X.iPtr, cuind))
            X
        end
        function gthr(X::CudaSparseVector{$elty},
                      Y::CudaVector{$elty},
                      index::SparseChar)
            gthr!(copy(X),Y,index)
        end
    end
end

for (fname,elty) in ((:cusparseSgthrz, :Float32),
                     (:cusparseDgthrz, :Float64),
                     (:cusparseCgthrz, :Complex64),
                     (:cusparseZgthrz, :Complex128))
    @eval begin
        function gthrz!(X::CudaSparseVector{$elty},
                        Y::CudaVector{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{$elty},
                               Ptr{Cint}, cusparseIndexBase_t), cusparsehandle[1],
                              X.nnz, Y, X.nzVal, X.iPtr, cuind))
            X,Y
        end
        function gthrz(X::CudaSparseVector{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            gthrz!(copy(X),copy(Y),index)
        end
    end
end

for (fname,elty) in ((:cusparseSroti, :Float32),
                     (:cusparseDroti, :Float64))
    @eval begin
        function roti!(X::CudaSparseVector{$elty},
                       Y::CudaVector{$elty},
                       c::$elty,
                       s::$elty,
                       index::SparseChar)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{$Cint},
                               Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, cusparseIndexBase_t),
                              cusparsehandle[1], X.nnz, X.nzVal, X.iPtr, Y, [c], [s], cuind))
            X,Y
        end
        function roti(X::CudaSparseVector{$elty},
                      Y::CudaVector{$elty},
                      c::$elty,
                      s::$elty,
                      index::SparseChar)
            roti!(copy(X),copy(Y),c,s,index)
        end
    end
end

for (fname,elty) in ((:cusparseSsctr, :Float32),
                     (:cusparseDsctr, :Float64),
                     (:cusparseCsctr, :Complex64),
                     (:cusparseZsctr, :Complex128))
    @eval begin
        function sctr!(X::CudaSparseVector{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{Cint},
                               Ptr{$elty}, cusparseIndexBase_t),
                              cusparsehandle[1], X.nnz, X.nzVal, X.iPtr,
                              Y, cuind))
            Y
        end
        function sctr(X::CudaSparseVector{$elty},
                      index::SparseChar)
            sctr!(X,CudaArray(zeros($elty,X.dims[1])),index)
        end
    end
end

## level 2 functions

for (fname,elty) in ((:cusparseSbsrmv, :Float32),
                     (:cusparseDbsrmv, :Float64),
                     (:cusparseCbsrmv, :Complex64),
                     (:cusparseZbsrmv, :Complex128))
    @eval begin
        function mv!(transa::SparseChar,
                     alpha::$elty,
                     A::CudaSparseMatrixBSR{$elty},
                     X::CudaVector{$elty},
                     beta::$elty,
                     Y::CudaVector{$elty},
                     index::SparseChar)
            cudir = cusparsedir(A.dir)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            mb = div(m,A.blockDim)
            nb = div(n,A.blockDim)
            if transa == 'N'
                chkmvdims(X,n,Y,m)
            end
            if transa == 'T' || transa == 'C'
                chkmvdims(X,m,Y,n)
            end
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, Cint, Cint, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Cint,
                               Ptr{$elty}, Ptr{$elty}, Ptr{$elty}),
                              cusparsehandle[1], cudir, cutransa, mb, nb,
                              A.nnz, [alpha], &cudesc, A.nzVal, A.rowPtr,
                              A.colVal, A.blockDim, X, [beta], Y))
            Y
        end
    end
end

for (fname,elty) in ((:cusparseScsrmv, :Float32),
                     (:cusparseDcsrmv, :Float64),
                     (:cusparseCcsrmv, :Complex64),
                     (:cusparseZcsrmv, :Complex128))
    @eval begin
        function mv!(transa::SparseChar,
                     alpha::$elty,
                     A::Union{CudaSparseMatrixCSR{$elty},HermOrSym{$elty,CudaSparseMatrixCSR{$elty}}},
                     X::CudaVector{$elty},
                     beta::$elty,
                     Y::CudaVector{$elty},
                     index::SparseChar)
            Mat     = A
            if typeof(A) <: Base.LinAlg.HermOrSym
                 Mat = A.data
            end
            cutransa = cusparseop(transa)
            m,n = Mat.dims
            if transa == 'N'
                chkmvdims(X, n, Y, m)
            end
            if transa == 'T' || transa == 'C'
                chkmvdims(X, m, Y, n)
            end
            cudesc = getDescr(A,index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                               Ptr{$elty}, Ptr{$elty}), cusparsehandle[1],
                               cutransa, m, n, Mat.nnz, [alpha], &cudesc, Mat.nzVal,
                               Mat.rowPtr, Mat.colVal, X, [beta], Y))
            Y
        end
        function mv!(transa::SparseChar,
                     alpha::$elty,
                     A::Union{CudaSparseMatrixCSC{$elty},HermOrSym{$elty,CudaSparseMatrixCSC{$elty}}},
                     X::CudaVector{$elty},
                     beta::$elty,
                     Y::CudaVector{$elty},
                     index::SparseChar)
            Mat     = A
            if typeof(A) <: Base.LinAlg.HermOrSym
                 Mat = A.data
            end
            ctransa  = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cudesc   = getDescr(A,index)
            n,m      = Mat.dims
            if ctransa == 'N'
                chkmvdims(X,n,Y,m)
            end
            if ctransa == 'T' || ctransa == 'C'
                chkmvdims(X,m,Y,n)
            end
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                               Ptr{$elty}, Ptr{$elty}), cusparsehandle[1],
                               cutransa, m, n, Mat.nnz, [alpha], &cudesc,
                               Mat.nzVal, Mat.colPtr, Mat.rowVal, X, [beta], Y))
            Y
        end
    end
end

# bsrsv2
for (bname,aname,sname,elty) in ((:cusparseSbsrsv2_bufferSize, :cusparseSbsrsv2_analysis, :cusparseSbsrsv2_solve, :Float32),
                                 (:cusparseDbsrsv2_bufferSize, :cusparseDbsrsv2_analysis, :cusparseDbsrsv2_solve, :Float64),
                                 (:cusparseCbsrsv2_bufferSize, :cusparseCbsrsv2_analysis, :cusparseCbsrsv2_solve, :Complex64),
                                 (:cusparseZbsrsv2_bufferSize, :cusparseZbsrsv2_analysis, :cusparseZbsrsv2_solve, :Complex128))
    @eval begin
        function sv2!(transa::SparseChar,
                      uplo::SparseChar,
                      alpha::$elty,
                      A::CudaSparseMatrixBSR{$elty},
                      X::CudaVector{$elty},
                      index::SparseChar)
            cutransa = cusparseop(transa)
            cudir    = cusparsedir(A.dir)
            cuind    = cusparseindex(index)
            cuplo    = cusparsefill(uplo)
            cudesc   = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, cuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n      = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            mX = length(X)
            if( mX != m )
                throw(DimensionMismatch("X must have length $m, but has length $mX"))
            end
            info = bsrsv2Info_t[0]
            cusparseCreateBsrsv2Info(info)
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Cint, bsrsv2Info_t, Ptr{Cint}),
                              cusparsehandle[1], cudir, cutransa, mb, A.nnz,
                              &cudesc, A.nzVal, A.rowPtr, A.colVal,
                              A.blockDim, info[1], bufSize))
            buffer = CudaArray(zeros(UInt8, bufSize[1]))
            statuscheck(ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Cint, bsrsv2Info_t,
                               cusparseSolvePolicy_t, Ptr{Void}),
                              cusparsehandle[1], cudir, cutransa, mb, A.nnz,
                              &cudesc, A.nzVal, A.rowPtr, A.colVal, A.blockDim,
                              info[1], CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            posit = Array(Cint,1)
            statuscheck(ccall((:cusparseXbsrsv2_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, bsrsv2Info_t,
                        Ptr{Cint}), cusparsehandle[1], info[1], posit))
            if( posit[1] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[1],posit[1],")"))
            end
            statuscheck(ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, Cint, Cint, Ptr{$elty},
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Cint, bsrsv2Info_t, Ptr{$elty},
                               Ptr{$elty}, cusparseSolvePolicy_t, Ptr{Void}),
                              cusparsehandle[1], cudir, cutransa, mb, A.nnz,
                              [alpha], &cudesc, A.nzVal, A.rowPtr, A.colVal,
                              A.blockDim, info[1], X, X,
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            cusparseDestroyBsrsv2Info(info[1])
            X
        end
    end
end

for elty in (:Float32, :Float64, :Complex64, :Complex128)
    @eval begin
        function sv2(transa::SparseChar,
                     uplo::SparseChar,
                     alpha::$elty,
                     A::CudaSparseMatrix{$elty},
                     X::CudaVector{$elty},
                     index::SparseChar)
            sv2!(transa,uplo,alpha,A,copy(X),index)
        end
        function sv2(transa::SparseChar,
                     uplo::SparseChar,
                     A::CudaSparseMatrix{$elty},
                     X::CudaVector{$elty},
                     index::SparseChar)
            sv2!(transa,uplo,one($elty),A,copy(X),index)
        end
        function sv2(transa::SparseChar,
                     alpha::$elty,
                     A::AbstractTriangular,
                     X::CudaVector{$elty},
                     index::SparseChar)
            uplo = 'U'
            if islower(A)
                uplo = 'L'
            end
            sv2!(transa,uplo,alpha,A.data,copy(X),index)
        end
        function sv2(transa::SparseChar,
                     A::AbstractTriangular,
                     X::CudaVector{$elty},
                     index::SparseChar)
            uplo = 'U'
            if islower(A)
                uplo = 'L'
            end
            sv2!(transa,uplo,one($elty),A.data,copy(X),index)
        end
    end
end

for (fname,elty) in ((:cusparseScsrsv_analysis, :Float32),
                     (:cusparseDcsrsv_analysis, :Float64),
                     (:cusparseCcsrsv_analysis, :Complex64),
                     (:cusparseZcsrsv_analysis, :Complex128))
    @eval begin
        function sv_analysis(transa::SparseChar,
                             typea::SparseChar,
                             uplo::SparseChar,
                             A::CudaSparseMatrixCSR{$elty},
                             index::SparseChar)
            cutransa = cusparseop(transa)
            cuind    = cusparseindex(index)
            cutype   = cusparsetype(typea)
            cuuplo   = cusparsefill(uplo)
            cudesc   = cusparseMatDescr_t(cutype, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = cusparseSolveAnalysisInfo_t[0]
            cusparseCreateSolveAnalysisInfo(info)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint},
                               cusparseSolveAnalysisInfo_t), cusparsehandle[1],
                               cutransa, m, A.nnz, &cudesc, A.nzVal,
                               A.rowPtr, A.colVal, info[1]))
            info[1]
        end
    end
end

#cscsv_analysis
for (fname,elty) in ((:cusparseScsrsv_analysis, :Float32),
                     (:cusparseDcsrsv_analysis, :Float64),
                     (:cusparseCcsrsv_analysis, :Complex64),
                     (:cusparseZcsrsv_analysis, :Complex128))
    @eval begin
        function sv_analysis(transa::SparseChar,
                             typea::SparseChar,
                             uplo::SparseChar,
                             A::CudaSparseMatrixCSC{$elty},
                             index::SparseChar)
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cuplo = 'U'
            if uplo == 'U'
                cuplo = 'L'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cutype   = cusparsetype(typea)
            cuuplo   = cusparsefill(cuplo)
            cudesc   = cusparseMatDescr_t(cutype, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            n,m      = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = cusparseSolveAnalysisInfo_t[0]
            cusparseCreateSolveAnalysisInfo(info)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint},
                               cusparseSolveAnalysisInfo_t), cusparsehandle[1],
                               cutransa, m, A.nnz, &cudesc, A.nzVal,
                               A.colPtr, A.rowVal, info[1]))
            info[1]
        end
    end
end

# csr solve
for (fname,elty) in ((:cusparseScsrsv_solve, :Float32),
                     (:cusparseDcsrsv_solve, :Float64),
                     (:cusparseCcsrsv_solve, :Complex64),
                     (:cusparseZcsrsv_solve, :Complex128))
    @eval begin
        function sv_solve!(transa::SparseChar,
                           uplo::SparseChar,
                           alpha::$elty,
                           A::CudaSparseMatrixCSR{$elty},
                           X::CudaVector{$elty},
                           Y::CudaVector{$elty},
                           info::cusparseSolveAnalysisInfo_t,
                           index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( size(X)[1] != m )
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, cusparseSolveAnalysisInfo_t,
                               Ptr{$elty}, Ptr{$elty}), cusparsehandle[1],
                               cutransa, m, [alpha], &cudesc, A.nzVal,
                               A.rowPtr, A.colVal, info, X, Y))
            Y
        end
    end
end

# csc solve
for (fname,elty) in ((:cusparseScsrsv_solve, :Float32),
                     (:cusparseDcsrsv_solve, :Float64),
                     (:cusparseCcsrsv_solve, :Complex64),
                     (:cusparseZcsrsv_solve, :Complex128))

    @eval begin
        function sv_solve!(transa::SparseChar,
                           uplo::SparseChar,
                           alpha::$elty,
                           A::CudaSparseMatrixCSC{$elty},
                           X::CudaVector{$elty},
                           Y::CudaVector{$elty},
                           info::cusparseSolveAnalysisInfo_t,
                           index::SparseChar)
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cuplo = 'U'
            if uplo == 'U'
                cuplo = 'L'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cuuplo   = cusparsefill(cuplo)
            cudesc   = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            n,m      = A.dims
            if( size(X)[1] != m )
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, cusparseSolveAnalysisInfo_t,
                               Ptr{$elty}, Ptr{$elty}), cusparsehandle[1],
                               cutransa, m, [alpha], &cudesc, A.nzVal,
                               A.colPtr, A.rowVal, info, X, Y))
            Y
        end
    end
end

# csrsv2
for (bname,aname,sname,elty) in ((:cusparseScsrsv2_bufferSize, :cusparseScsrsv2_analysis, :cusparseScsrsv2_solve, :Float32),
                                 (:cusparseDcsrsv2_bufferSize, :cusparseDcsrsv2_analysis, :cusparseDcsrsv2_solve, :Float64),
                                 (:cusparseCcsrsv2_bufferSize, :cusparseCcsrsv2_analysis, :cusparseCcsrsv2_solve, :Complex64),
                                 (:cusparseZcsrsv2_bufferSize, :cusparseZcsrsv2_analysis, :cusparseZcsrsv2_solve, :Complex128))
    @eval begin
        function sv2!(transa::SparseChar,
                      uplo::SparseChar,
                      alpha::$elty,
                      A::CudaSparseMatrixCSR{$elty},
                      X::CudaVector{$elty},
                      index::SparseChar)
            cutransa  = cusparseop(transa)
            cuind     = cusparseindex(index)
            cuuplo    = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mX = length(X)
            if( mX != m )
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            info = csrsv2Info_t[0]
            cusparseCreateCsrsv2Info(info)
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csrsv2Info_t, Ptr{Cint}),
                              cusparsehandle[1], cutransa, m, A.nnz,
                              &cudesc, A.nzVal, A.rowPtr, A.colVal,
                              info[1], bufSize))
            buffer = CudaArray(zeros(UInt8, bufSize[1]))
            statuscheck(ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csrsv2Info_t, cusparseSolvePolicy_t,
                               Ptr{Void}), cusparsehandle[1], cutransa, m, A.nnz,
                               &cudesc, A.nzVal, A.rowPtr, A.colVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            posit = Array(Cint,1)
            statuscheck(ccall((:cusparseXcsrsv2_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, csrsv2Info_t,
                        Ptr{Cint}), cusparsehandle[1], info[1], posit))
            if( posit[1] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[1],posit[1],")"))
            end
            statuscheck(ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, csrsv2Info_t,
                               Ptr{$elty}, Ptr{$elty}, cusparseSolvePolicy_t,
                               Ptr{Void}), cusparsehandle[1], cutransa, m,
                               A.nnz, [alpha], &cudesc, A.nzVal, A.rowPtr,
                               A.colVal, info[1], X, X,
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            cusparseDestroyCsrsv2Info(info[1])
            X
        end
    end
end

# cscsv2
for (bname,aname,sname,elty) in ((:cusparseScsrsv2_bufferSize, :cusparseScsrsv2_analysis, :cusparseScsrsv2_solve, :Float32),
                                 (:cusparseDcsrsv2_bufferSize, :cusparseDcsrsv2_analysis, :cusparseDcsrsv2_solve, :Float64),
                                 (:cusparseCcsrsv2_bufferSize, :cusparseCcsrsv2_analysis, :cusparseCcsrsv2_solve, :Complex64),
                                 (:cusparseZcsrsv2_bufferSize, :cusparseZcsrsv2_analysis, :cusparseZcsrsv2_solve, :Complex128))
    @eval begin
        function sv2!(transa::SparseChar,
                      uplo::SparseChar,
                      alpha::$elty,
                      A::CudaSparseMatrixCSC{$elty},
                      X::CudaVector{$elty},
                      index::SparseChar)
            ctransa = 'N'
            cuplo = 'U'
            if transa == 'N'
                ctransa = 'T'
            end
            if uplo == 'U'
                cuplo = 'L'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cuuplo   = cusparsefill(cuplo)
            cudesc   = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            n,m      = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mX = length(X)
            if( mX != m )
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            info = csrsv2Info_t[0]
            cusparseCreateCsrsv2Info(info)
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csrsv2Info_t, Ptr{Cint}),
                              cusparsehandle[1], cutransa, m, A.nnz,
                              &cudesc, A.nzVal, A.colPtr, A.rowVal,
                              info[1], bufSize))
            buffer = CudaArray(zeros(UInt8, bufSize[1]))
            statuscheck(ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csrsv2Info_t, cusparseSolvePolicy_t,
                               Ptr{Void}), cusparsehandle[1], cutransa, m, A.nnz,
                               &cudesc, A.nzVal, A.colPtr, A.rowVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            posit = Array(Cint,1)
            statuscheck(ccall((:cusparseXcsrsv2_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, csrsv2Info_t,
                        Ptr{Cint}), cusparsehandle[1], info[1], posit))
            if( posit[1] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[1],posit[1],")"))
            end
            statuscheck(ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, csrsv2Info_t,
                               Ptr{$elty}, Ptr{$elty}, cusparseSolvePolicy_t,
                               Ptr{Void}), cusparsehandle[1], cutransa, m,
                               A.nnz, [alpha], &cudesc, A.nzVal, A.colPtr,
                               A.rowVal, info[1], X, X,
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            cusparseDestroyCsrsv2Info(info[1])
            X
        end
    end
end

(\)(A::AbstractTriangular,B::CudaVector)       = sv2('N',A,B,'O')
At_ldiv_B(A::AbstractTriangular,B::CudaVector) = sv2('T',A,B,'O')
Ac_ldiv_B(A::AbstractTriangular,B::CudaVector) = sv2('C',A,B,'O')
(\){T}(A::AbstractTriangular{T,CudaSparseMatrixHYB{T}},B::CudaVector{T})       = sv('N',A,B,'O')
At_ldiv_B{T}(A::AbstractTriangular{T,CudaSparseMatrixHYB{T}},B::CudaVector{T}) = sv('T',A,B,'O')
Ac_ldiv_B{T}(A::AbstractTriangular{T,CudaSparseMatrixHYB{T}},B::CudaVector{T}) = sv('C',A,B,'O')

for (fname,elty) in ((:cusparseShybmv, :Float32),
                     (:cusparseDhybmv, :Float64),
                     (:cusparseChybmv, :Complex64),
                     (:cusparseZhybmv, :Complex128))
    @eval begin
        function mv!(transa::SparseChar,
                     alpha::$elty,
                     A::CudaSparseMatrixHYB{$elty},
                     X::CudaVector{$elty},
                     beta::$elty,
                     Y::CudaVector{$elty},
                     index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if transa == 'N'
                chkmvdims(X,n,Y,m)
            end
            if transa == 'T' || transa == 'C'
                chkmvdims(X,m,Y,n)
            end
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               cusparseHybMat_t, Ptr{$elty},
                               Ptr{$elty}, Ptr{$elty}), cusparsehandle[1],
                               cutransa, [alpha], &cudesc, A.Mat, X, [beta], Y))
            Y
        end
    end
end

for elty in (:Float32, :Float64, :Complex64, :Complex128)
    @eval begin
        function mv(transa::SparseChar,
                    alpha::$elty,
                    A::Union{CudaSparseMatrix{$elty},CompressedSparse{$elty}},
                    X::CudaVector{$elty},
                    beta::$elty,
                    Y::CudaVector{$elty},
                    index::SparseChar)
            mv!(transa,alpha,A,X,beta,copy(Y),index)
        end
        function mv(transa::SparseChar,
                    alpha::$elty,
                    A::Union{CudaSparseMatrix{$elty},CompressedSparse{$elty}},
                    X::CudaVector{$elty},
                    Y::CudaVector{$elty},
                    index::SparseChar)
            mv(transa,alpha,A,X,one($elty),Y,index)
        end
        function mv(transa::SparseChar,
                    A::Union{CudaSparseMatrix{$elty},CompressedSparse{$elty}},
                    X::CudaVector{$elty},
                    beta::$elty,
                    Y::CudaVector{$elty},
                    index::SparseChar)
            mv(transa,one($elty),A,X,beta,Y,index)
        end
        function mv(transa::SparseChar,
                    A::Union{CudaSparseMatrix{$elty},CompressedSparse{$elty}},
                    X::CudaVector{$elty},
                    Y::CudaVector{$elty},
                    index::SparseChar)
            mv(transa,one($elty),A,X,one($elty),Y,index)
        end
        function mv(transa::SparseChar,
                    alpha::$elty,
                    A::Union{CudaSparseMatrix{$elty},CompressedSparse{$elty}},
                    X::CudaVector{$elty},
                    index::SparseChar)
            mv(transa,alpha,A,X,zero($elty),CudaArray(zeros($elty,size(A)[1])),index)
        end
        function mv(transa::SparseChar,
                    A::Union{CudaSparseMatrix{$elty},CompressedSparse{$elty}},
                    X::CudaVector{$elty},
                    index::SparseChar)
            mv(transa,one($elty),A,X,zero($elty),CudaArray(zeros($elty,size(A)[1])),index)
        end
    end
end

(*){T}(A::CudaSparseMatrix{T},B::CudaMatrix{T})       = mm2('N','N',A,B,'O')
A_mul_Bt{T}(A::CudaSparseMatrix{T},B::CudaMatrix{T})  = mm2('N','T',A,B,'O')
At_mul_B{T}(A::CudaSparseMatrix{T},B::CudaMatrix{T})  = mm2('T','N',A,B,'O')
At_mul_Bt{T}(A::CudaSparseMatrix{T},B::CudaMatrix{T}) = mm2('T','T',A,B,'O')
Ac_mul_B{T}(A::CudaSparseMatrix{T},B::CudaMatrix{T})  = mm2('C','N',A,B,'O')

(*)(A::HermOrSym,B::CudaMatrix) = mm('N',A,B,'O')
At_mul_B(A::HermOrSym,B::CudaMatrix) = mm('T',A,B,'O')
Ac_mul_B(A::HermOrSym,B::CudaMatrix) = mm('C',A,B,'O')

for (fname,elty) in ((:cusparseShybsv_analysis, :Float32),
                     (:cusparseDhybsv_analysis, :Float64),
                     (:cusparseChybsv_analysis, :Complex64),
                     (:cusparseZhybsv_analysis, :Complex128))
    @eval begin
        function sv_analysis(transa::SparseChar,
                             typea::SparseChar,
                             uplo::SparseChar,
                             A::CudaSparseMatrixHYB{$elty},
                             index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = cusparseSolveAnalysisInfo_t[0]
            cusparseCreateSolveAnalysisInfo(info)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               Ptr{cusparseMatDescr_t}, cusparseHybMat_t,
                               cusparseSolveAnalysisInfo_t),
                              cusparsehandle[1], cutransa, &cudesc, A.Mat,
                              info[1]))
            info[1]
        end
    end
end

for (fname,elty) in ((:cusparseShybsv_solve, :Float32),
                     (:cusparseDhybsv_solve, :Float64),
                     (:cusparseChybsv_solve, :Complex64),
                     (:cusparseZhybsv_solve, :Complex128))
    @eval begin
        function sv_solve!(transa::SparseChar,
                           uplo::SparseChar,
                           alpha::$elty,
                           A::CudaSparseMatrixHYB{$elty},
                           X::CudaVector{$elty},
                           Y::CudaVector{$elty},
                           info::cusparseSolveAnalysisInfo_t,
                           index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( size(X)[1] != m )
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               cusparseHybMat_t, cusparseSolveAnalysisInfo_t,
                               Ptr{$elty}, Ptr{$elty}), cusparsehandle[1],
                               cutransa, [alpha], &cudesc, A.Mat, info, X, Y))
            Y
        end
    end
end

for elty in (:Float32, :Float64, :Complex64, :Complex128)
    @eval begin
        function sv_solve(transa::SparseChar,
                          uplo::SparseChar,
                          alpha::$elty,
                          A::CudaSparseMatrix{$elty},
                          X::CudaVector{$elty},
                          info::cusparseSolveAnalysisInfo_t,
                          index::SparseChar)
            Y = similar(X)
            sv_solve!(transa, uplo, alpha, A, X, Y, info, index)
        end
        function sv(transa::SparseChar,
                    typea::SparseChar,
                    uplo::SparseChar,
                    alpha::$elty,
                    A::CudaSparseMatrix{$elty},
                    X::CudaVector{$elty},
                    index::SparseChar)
            info = sv_analysis(transa,typea,uplo,A,index)
            sv_solve(transa,uplo,alpha,A,X,info,index)
        end
        function sv(transa::SparseChar,
                    typea::SparseChar,
                    uplo::SparseChar,
                    A::CudaSparseMatrix{$elty},
                    X::CudaVector{$elty},
                    index::SparseChar)
            info = sv_analysis(transa,typea,uplo,A,index)
            sv_solve(transa,uplo,one($elty),A,X,info,index)
        end
        function sv(transa::SparseChar,
                    A::AbstractTriangular,
                    X::CudaVector{$elty},
                    index::SparseChar)
            uplo = 'U'
            if islower(A)
                uplo = 'L'
            end
            info = sv_analysis(transa,'T',uplo,A.data,index)
            sv_solve(transa,uplo,one($elty),A.data,X,info,index)
        end
        function sv_analysis(transa::SparseChar,
                             typea::SparseChar,
                             uplo::SparseChar,
                             A::HermOrSym{$elty},
                             index::SparseChar)
            sv_analysis(transa,typea,uplo,A.data,index)
        end
    end
end
## level 3 functions

# bsrmm
for (fname,elty) in ((:cusparseSbsrmm, :Float32),
                     (:cusparseDbsrmm, :Float64),
                     (:cusparseCbsrmm, :Complex64),
                     (:cusparseZbsrmm, :Complex128))
    @eval begin
        function mm2!(transa::SparseChar,
                      transb::SparseChar,
                      alpha::$elty,
                      A::CudaSparseMatrixBSR{$elty},
                      B::CudaMatrix{$elty},
                      beta::$elty,
                      C::CudaMatrix{$elty},
                      index::SparseChar)
            cutransa = cusparseop(transa)
            cutransb = cusparseop(transb)
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,k = A.dims
            mb = div(m,A.blockDim)
            kb = div(k,A.blockDim)
            n = size(C)[2]
            if transa == 'N' && transb == 'N'
                chkmmdims(B,C,k,n,m,n)
            elseif transa == 'N' && transb != 'N'
                chkmmdims(B,C,n,k,m,n)
            elseif transa != 'N' && transb == 'N'
                chkmmdims(B,C,m,n,k,n)
            elseif transa != 'N' && transb != 'N'
                chkmmdims(B,C,n,m,k,n)
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, cusparseOperation_t, Cint,
                               Cint, Cint, Cint, Ptr{$elty},
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Cint, Ptr{$elty}, Cint, Ptr{$elty},
                               Ptr{$elty}, Cint), cusparsehandle[1], cudir,
                               cutransa, cutransb, mb, n, kb, A.nnz,
                               [alpha], &cudesc, A.nzVal,A.rowPtr, A.colVal,
                               A.blockDim, B, ldb, [beta], C, ldc))
            C
        end
    end
end

# csrmm
for (fname,elty) in ((:cusparseScsrmm, :Float32),
                     (:cusparseDcsrmm, :Float64),
                     (:cusparseCcsrmm, :Complex64),
                     (:cusparseZcsrmm, :Complex128))
    @eval begin
        function mm!(transa::SparseChar,
                     alpha::$elty,
                     A::Union{HermOrSym{$elty,CudaSparseMatrixCSR{$elty}},CudaSparseMatrixCSR{$elty}},
                     B::CudaMatrix{$elty},
                     beta::$elty,
                     C::CudaMatrix{$elty},
                     index::SparseChar)
            Mat     = A
            if typeof(A) <: Base.LinAlg.HermOrSym
                 Mat = A.data
            end
            cutransa = cusparseop(transa)
            cuind    = cusparseindex(index)
            cudesc   = getDescr(A,index)
            m,k      = Mat.dims
            n        = size(C)[2]
            if transa == 'N'
                chkmmdims(B,C,k,n,m,n)
            else
                chkmmdims(B,C,m,n,k,n)
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                               Cint, Ptr{$elty}, Ptr{$elty}, Cint),
                               cusparsehandle[1], cutransa, m, n, k, Mat.nnz,
                               [alpha], &cudesc, Mat.nzVal, Mat.rowPtr,
                               Mat.colVal, B, ldb, [beta], C, ldc))
            C
        end
        function mm!(transa::SparseChar,
                     alpha::$elty,
                     A::Union{HermOrSym{$elty,CudaSparseMatrixCSC{$elty}},CudaSparseMatrixCSC{$elty}},
                     B::CudaMatrix{$elty},
                     beta::$elty,
                     C::CudaMatrix{$elty},
                     index::SparseChar)
            Mat     = A
            if typeof(A) <: Base.LinAlg.HermOrSym
                Mat = A.data
            end
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cudesc   = getDescr(A,index)
            k,m      = Mat.dims
            n        = size(C)[2]
            if ctransa == 'N'
                chkmmdims(B,C,k,n,m,n)
            else
                chkmmdims(B,C,m,n,k,n)
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                               Cint, Ptr{$elty}, Ptr{$elty}, Cint),
                               cusparsehandle[1], cutransa, m, n, k, Mat.nnz,
                               [alpha], &cudesc, Mat.nzVal, Mat.colPtr,
                               Mat.rowVal, B, ldb, [beta], C, ldc))
            C
        end
    end
end

for elty in (:Float32, :Float64, :Complex64, :Complex128)
    @eval begin
        function mm(transa::SparseChar,
                    alpha::$elty,
                    A::CudaSparseMatrix{$elty},
                    B::CudaMatrix{$elty},
                    beta::$elty,
                    C::CudaMatrix{$elty},
                    index::SparseChar)
            mm!(transa,alpha,A,B,beta,copy(C),index)
        end
        function mm(transa::SparseChar,
                    A::CudaSparseMatrix{$elty},
                    B::CudaMatrix{$elty},
                    beta::$elty,
                    C::CudaMatrix{$elty},
                    index::SparseChar)
            mm(transa,one($elty),A,B,beta,C,index)
        end
        function mm(transa::SparseChar,
                    A::CudaSparseMatrix{$elty},
                    B::CudaMatrix{$elty},
                    C::CudaMatrix{$elty},
                    index::SparseChar)
            mm(transa,one($elty),A,B,one($elty),C,index)
        end
        function mm(transa::SparseChar,
                    alpha::$elty,
                    A::CudaSparseMatrix{$elty},
                    B::CudaMatrix{$elty},
                    index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            mm!(transa,alpha,A,B,zero($elty),CudaArray(zeros($elty,(m,size(B)[2]))),index)
        end
        function mm(transa::SparseChar,
                    A::CudaSparseMatrix{$elty},
                    B::CudaMatrix{$elty},
                    index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            mm!(transa,one($elty),A,B,zero($elty),CudaArray(zeros($elty,(m,size(B)[2]))),index)
        end
        function mm(transa::SparseChar,
                    A::HermOrSym,
                    B::CudaMatrix{$elty},
                    index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            mm!(transa,one($elty),A.data,B,zero($elty),CudaArray(zeros($elty,(m,size(B)[2]))),index)
        end
    end
end

for (fname,elty) in ((:cusparseScsrmm2, :Float32),
                     (:cusparseDcsrmm2, :Float64),
                     (:cusparseCcsrmm2, :Complex64),
                     (:cusparseZcsrmm2, :Complex128))
    @eval begin
        function mm2!(transa::SparseChar,
                      transb::SparseChar,
                      alpha::$elty,
                      A::CudaSparseMatrixCSR{$elty},
                      B::CudaMatrix{$elty},
                      beta::$elty,
                      C::CudaMatrix{$elty},
                      index::SparseChar)
            cutransa = cusparseop(transa)
            cutransb = cusparseop(transb)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,k = A.dims
            n = size(C)[2]
            if transa == 'N' && transb == 'N'
                chkmmdims(B,C,k,n,m,n)
            elseif transa == 'N' && transb != 'N'
                chkmmdims(B,C,n,k,m,n)
            elseif transa != 'N' && transb == 'N'
                chkmmdims(B,C,m,n,k,n)
            elseif transa != 'N' && transb != 'N'
                chkmmdims(B,C,n,m,k,n)
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Cint,
                               Ptr{$elty}, Ptr{$elty}, Cint), cusparsehandle[1],
                               cutransa, cutransb, m, n, k, A.nnz, [alpha], &cudesc,
                               A.nzVal, A.rowPtr, A.colVal, B, ldb, [beta], C, ldc))
            C
        end
        function mm2!(transa::SparseChar,
                      transb::SparseChar,
                      alpha::$elty,
                      A::CudaSparseMatrixCSC{$elty},
                      B::CudaMatrix{$elty},
                      beta::$elty,
                      C::CudaMatrix{$elty},
                      index::SparseChar)
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cutransa = cusparseop(ctransa)
            cutransb = cusparseop(transb)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            k,m = A.dims
            n = size(C)[2]
            if ctransa == 'N' && transb == 'N'
                chkmmdims(B,C,k,n,m,n)
            elseif ctransa == 'N' && transb != 'N'
                chkmmdims(B,C,n,k,m,n)
            elseif ctransa != 'N' && transb == 'N'
                chkmmdims(B,C,m,n,k,n)
            elseif ctransa != 'N' && transb != 'N'
                chkmmdims(B,C,n,m,k,n)
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Cint,
                               Ptr{$elty}, Ptr{$elty}, Cint), cusparsehandle[1],
                               cutransa, cutransb, m, n, k, A.nnz, [alpha], &cudesc,
                               A.nzVal, A.colPtr, A.rowVal, B, ldb, [beta], C, ldc))
            C
        end
    end
end

for elty in (:Float32,:Float64,:Complex64,:Complex128)
    @eval begin
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     alpha::$elty,
                     A::Union{CudaSparseMatrixCSR{$elty},CudaSparseMatrixCSC{$elty},CudaSparseMatrixBSR{$elty}},
                     B::CudaMatrix{$elty},
                     beta::$elty,
                     C::CudaMatrix{$elty},
                     index::SparseChar)
            mm2!(transa,transb,alpha,A,B,beta,copy(C),index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     A::Union{CudaSparseMatrixCSR{$elty},CudaSparseMatrixCSC{$elty},CudaSparseMatrixBSR{$elty}},
                     B::CudaMatrix{$elty},
                     beta::$elty,
                     C::CudaMatrix{$elty},
                     index::SparseChar)
            mm2(transa,transb,one($elty),A,B,beta,C,index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     A::Union{CudaSparseMatrixCSR{$elty},CudaSparseMatrixCSC{$elty},CudaSparseMatrixBSR{$elty}},
                     B::CudaMatrix{$elty},
                     C::CudaMatrix{$elty},
                     index::SparseChar)
            mm2(transa,transb,one($elty),A,B,one($elty),C,index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     alpha::$elty,
                     A::Union{CudaSparseMatrixCSR{$elty},CudaSparseMatrixCSC{$elty},CudaSparseMatrixBSR{$elty}},
                     B::CudaMatrix{$elty},
                     index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            n = transb == 'N' ? size(B)[2] : size(B)[1]
            mm2(transa,transb,alpha,A,B,zero($elty),CudaArray(zeros($elty,(m,n))),index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     A::Union{CudaSparseMatrixCSR{$elty},CudaSparseMatrixCSC{$elty},CudaSparseMatrixBSR{$elty}},
                     B::CudaMatrix{$elty},
                     index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            n = transb == 'N' ? size(B)[2] : size(B)[1]
            mm2(transa,transb,one($elty),A,B,zero($elty),CudaArray(zeros($elty,(m,n))),index)
        end
    end
end

(*)(A::CudaSparseMatrix,B::CudaVector)       = mv('N',A,B,'O')
At_mul_B(A::CudaSparseMatrix,B::CudaVector)  = mv('T',A,B,'O')
Ac_mul_B(A::CudaSparseMatrix,B::CudaVector)  = mv('C',A,B,'O')
(*){T}(A::HermOrSym{T,CudaSparseMatrix{T}},B::CudaVector{T}) = mv('N',A,B,'O')
At_mul_B{T}(A::HermOrSym{T,CudaSparseMatrix{T}},B::CudaVector{T}) = mv('T',A,B,'O')
Ac_mul_B{T}(A::HermOrSym{T,CudaSparseMatrix{T}},B::CudaVector{T}) = mv('C',A,B,'O')

for (fname,elty) in ((:cusparseScsrsm_analysis, :Float32),
                     (:cusparseDcsrsm_analysis, :Float64),
                     (:cusparseCcsrsm_analysis, :Complex64),
                     (:cusparseZcsrsm_analysis, :Complex128))
    @eval begin
        function sm_analysis(transa::SparseChar,
                             uplo::SparseChar,
                             A::CudaSparseMatrixCSR{$elty},
                             index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( n != m )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = cusparseSolveAnalysisInfo_t[0]
            cusparseCreateSolveAnalysisInfo(info)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, cusparseSolveAnalysisInfo_t),
                              cusparsehandle[1], cutransa, m, A.nnz, &cudesc,
                              A.nzVal, A.rowPtr, A.colVal, info[1]))
            info[1]
        end
        function sm_analysis(transa::SparseChar,
                             uplo::SparseChar,
                             A::CudaSparseMatrixCSC{$elty},
                             index::SparseChar)
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cuplo = 'U'
            if uplo == 'U'
                cuplo = 'L'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cuuplo   = cusparsefill(cuplo)
            cudesc   = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            n,m      = A.dims
            if( n != m )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = cusparseSolveAnalysisInfo_t[0]
            cusparseCreateSolveAnalysisInfo(info)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, cusparseSolveAnalysisInfo_t),
                              cusparsehandle[1], cutransa, m, A.nnz, &cudesc,
                              A.nzVal, A.colPtr, A.rowVal, info[1]))
            info[1]
        end
    end
end

for (fname,elty) in ((:cusparseScsrsm_solve, :Float32),
                     (:cusparseDcsrsm_solve, :Float64),
                     (:cusparseCcsrsm_solve, :Complex64),
                     (:cusparseZcsrsm_solve, :Complex128))
    @eval begin
        function sm_solve(transa::SparseChar,
                          uplo::SparseChar,
                          alpha::$elty,
                          A::CudaSparseMatrixCSR{$elty},
                          X::CudaMatrix{$elty},
                          info::cusparseSolveAnalysisInfo_t,
                          index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,nA = A.dims
            mX,n = X.dims
            if( mX != m )
                throw(DimensionMismatch("First dimension of A, $m, and X, $mX must match"))
            end
            Y = similar(X)
            ldx = max(1,stride(X,2))
            ldy = max(1,stride(Y,2))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, cusparseSolveAnalysisInfo_t,
                               Ptr{$elty}, Cint, Ptr{$elty}, Cint),
                              cusparsehandle[1], cutransa, m, n, [alpha],
                              &cudesc, A.nzVal, A.rowPtr, A.colVal, info, X, ldx,
                              Y, ldy))
            Y
        end
        function sm_solve(transa::SparseChar,
                          uplo::SparseChar,
                          alpha::$elty,
                          A::CudaSparseMatrixCSC{$elty},
                          X::CudaMatrix{$elty},
                          info::cusparseSolveAnalysisInfo_t,
                          index::SparseChar)
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cuplo = 'U'
            if uplo == 'U'
                cuplo = 'L'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cuuplo   = cusparsefill(cuplo)
            cudesc   = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,nA     = A.dims
            mX,n     = X.dims
            if( mX != m )
                throw(DimensionMismatch("First dimension of A, $m, and X, $mX must match"))
            end
            Y = similar(X)
            ldx = max(1,stride(X,2))
            ldy = max(1,stride(Y,2))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, cusparseSolveAnalysisInfo_t,
                               Ptr{$elty}, Cint, Ptr{$elty}, Cint),
                              cusparsehandle[1], cutransa, m, n, [alpha],
                              &cudesc, A.nzVal, A.colPtr, A.rowVal, info, X, ldx,
                              Y, ldy))
            Y
        end
    end
end

for elty in (:Float32, :Float64, :Complex64, :Complex128)
    @eval begin
        function sm(transa::SparseChar,
                    uplo::SparseChar,
                    alpha::$elty,
                    A::CudaSparseMatrix{$elty},
                    B::CudaMatrix{$elty},
                    index::SparseChar)
            info = sm_analysis(transa,uplo,A,index)
            sm_solve(transa,uplo,alpha,A,B,info,index)
        end
        function sm(transa::SparseChar,
                    uplo::SparseChar,
                    A::CudaSparseMatrix{$elty},
                    B::CudaMatrix{$elty},
                    index::SparseChar)
            info = sm_analysis(transa,uplo,A,index)
            sm_solve(transa,uplo,one($elty),A,B,info,index)
        end
        function sm(transa::SparseChar,
                    alpha::$elty,
                    A::AbstractTriangular,
                    B::CudaMatrix{$elty},
                    index::SparseChar)
            uplo = 'U'
            if islower(A)
                uplo = 'L'
            end
            info = sm_analysis(transa,uplo,A.data,index)
            sm_solve(transa,uplo,alpha,A.data,B,info,index)
        end
        function sm(transa::SparseChar,
                    A::AbstractTriangular,
                    B::CudaMatrix{$elty},
                    index::SparseChar)
            uplo = 'U'
            if islower(A)
                uplo = 'L'
            end
            info = sm_analysis(transa,uplo,A.data,index)
            sm_solve(transa,uplo,one($elty),A.data,B,info,index)
        end
    end
end

(\)(A::AbstractTriangular,B::CudaMatrix)       = sm('N',A,B,'O')
At_ldiv_B(A::AbstractTriangular,B::CudaMatrix) = sm('T',A,B,'O')
Ac_ldiv_B(A::AbstractTriangular,B::CudaMatrix) = sm('C',A,B,'O')

# bsrsm2
for (bname,aname,sname,elty) in ((:cusparseSbsrsm2_bufferSize, :cusparseSbsrsm2_analysis, :cusparseSbsrsm2_solve, :Float32),
                                 (:cusparseDbsrsm2_bufferSize, :cusparseDbsrsm2_analysis, :cusparseDbsrsm2_solve, :Float64),
                                 (:cusparseCbsrsm2_bufferSize, :cusparseCbsrsm2_analysis, :cusparseCbsrsm2_solve, :Complex64),
                                 (:cusparseZbsrsm2_bufferSize, :cusparseZbsrsm2_analysis, :cusparseZbsrsm2_solve, :Complex128))
    @eval begin
        function bsrsm2!(transa::SparseChar,
                         transxy::SparseChar,
                         alpha::$elty,
                         A::CudaSparseMatrixBSR{$elty},
                         X::CudaMatrix{$elty},
                         index::SparseChar)
            cutransa  = cusparseop(transa)
            cutransxy = cusparseop(transxy)
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square!"))
            end
            mb = div(m,A.blockDim)
            mX,nX = size(X)
            if( transxy == 'N' && (mX != m) )
                throw(DimensionMismatch(""))
            end
            if( transxy != 'N' && (nX != m) )
                throw(DimensionMismatch(""))
            end
            ldx = max(1,stride(X,2))
            info = bsrsm2Info_t[0]
            cusparseCreateBsrsm2Info(info)
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, cusparseOperation_t, Cint,
                               Cint, Cint, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Cint,
                               bsrsm2Info_t, Ptr{Cint}), cusparsehandle[1],
                               cudir, cutransa, cutransxy, mb, nX, A.nnz,
                               &cudesc, A.nzVal, A.rowPtr, A.colVal,
                               A.blockDim, info[1], bufSize))
            buffer = CudaArray(zeros(UInt8, bufSize[1]))
            statuscheck(ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, cusparseOperation_t, Cint,
                               Cint, Cint, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Cint,
                               bsrsm2Info_t, cusparseSolvePolicy_t, Ptr{Void}),
                              cusparsehandle[1], cudir, cutransa, cutransxy,
                              mb, nX, A.nnz, &cudesc, A.nzVal, A.rowPtr,
                              A.colVal, A.blockDim, info[1],
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            posit = Array(Cint,1)
            statuscheck(ccall((:cusparseXbsrsm2_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, bsrsm2Info_t,
                        Ptr{Cint}), cusparsehandle[1], info[1], posit))
            if( posit[1] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[1],posit[1],")"))
            end
            statuscheck(ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, cusparseOperation_t, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Cint,
                               bsrsm2Info_t, Ptr{$elty}, Cint, Ptr{$elty}, Cint,
                               cusparseSolvePolicy_t, Ptr{Void}),
                              cusparsehandle[1], cudir, cutransa, cutransxy, mb,
                              nX, A.nnz, [alpha], &cudesc, A.nzVal, A.rowPtr,
                              A.colVal, A.blockDim, info[1], X, ldx, X, ldx,
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            cusparseDestroyBsrsm2Info(info[1])
            X
        end
        function bsrsm2(transa::SparseChar,
                        transxy::SparseChar,
                        alpha::$elty,
                        A::CudaSparseMatrixBSR{$elty},
                        X::CudaMatrix{$elty},
                        index::SparseChar)
            bsrsm2!(transa,transxy,alpha,A,copy(X),index)
        end
    end
end

# extensions

# CSR GEAM
for (fname,elty) in ((:cusparseScsrgeam, :Float32),
                     (:cusparseDcsrgeam, :Float64),
                     (:cusparseCcsrgeam, :Complex64),
                     (:cusparseZcsrgeam, :Complex128))
    @eval begin
        function geam(alpha::$elty,
                      A::CudaSparseMatrixCSR{$elty},
                      beta::$elty,
                      B::CudaSparseMatrixCSR{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            cuinda = cusparseindex(indexA)
            cuindb = cusparseindex(indexB)
            cuindc = cusparseindex(indexB)
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescb = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindb)
            cudescc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            mA,nA = A.dims
            mB,nB = B.dims
            if( (mA != mB) || (nA != nB) )
                throw(DimensionMismatch(""))
            end
            nnzC = Array(Cint,1)
            rowPtrC = CudaArray(zeros(Cint,mA+1))
            statuscheck(ccall((:cusparseXcsrgeamNnz,libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint},
                               Ptr{Cint}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint},
                               Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{Cint},
                               Ptr{Cint}), cusparsehandle[1], mA, nA, &cudesca,
                               A.nnz, A.rowPtr, A.colVal, &cudescb, B.nnz,
                               B.rowPtr, B.colVal, &cudescc, rowPtrC, nnzC))
            nnz = nnzC[1]
            C = CudaSparseMatrixCSR(rowPtrC, CudaArray(zeros(Cint,nnz)), CudaArray(zeros($elty,nnz)), nnz, A.dims)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, Ptr{$elty},
                               Ptr{cusparseMatDescr_t}, Cint, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                               Ptr{cusparseMatDescr_t}, Cint, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}),
                              cusparsehandle[1], mA, nA, [alpha], &cudesca,
                              A.nnz, A.nzVal, A.rowPtr, A.colVal, [beta],
                              &cudescb, B.nnz, B.nzVal, B.rowPtr, B.colVal,
                              &cudescc, C.nzVal, C.rowPtr, C.colVal))
            C
        end
        function geam(alpha::$elty,
                      A::CudaSparseMatrixCSC{$elty},
                      beta::$elty,
                      B::CudaSparseMatrixCSC{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            cuinda = cusparseindex(indexA)
            cuindb = cusparseindex(indexB)
            cuindc = cusparseindex(indexB)
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescb = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindb)
            cudescc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            mA,nA = A.dims
            mB,nB = B.dims
            if( (mA != mB) || (nA != nB) )
                throw(DimensionMismatch("A and B must have same dimensions!"))
            end
            nnzC = Array(Cint,1)
            rowPtrC = CudaArray(zeros(Cint,mA+1))
            statuscheck(ccall((:cusparseXcsrgeamNnz,libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint},
                               Ptr{Cint}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint},
                               Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{Cint},
                               Ptr{Cint}), cusparsehandle[1], mA, nA, &cudesca,
                               A.nnz, A.colPtr, A.rowVal, &cudescb, B.nnz,
                               B.colPtr, B.rowVal, &cudescc, rowPtrC, nnzC))
            nnz = nnzC[1]
            C = CudaSparseMatrixCSC(rowPtrC, CudaArray(zeros(Cint,nnz)), CudaArray(zeros($elty,nnz)), nnz, A.dims)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, Ptr{$elty},
                               Ptr{cusparseMatDescr_t}, Cint, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                               Ptr{cusparseMatDescr_t}, Cint, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}),
                              cusparsehandle[1], mA, nA, [alpha], &cudesca,
                              A.nnz, A.nzVal, A.colPtr, A.rowVal, [beta],
                              &cudescb, B.nnz, B.nzVal, B.colPtr, B.rowVal,
                              &cudescc, C.nzVal, C.colPtr, C.rowVal))
            C
        end
        function geam(alpha::$elty,
                      A::CudaSparseMatrixCSR{$elty},
                      beta::$elty,
                      B::CudaSparseMatrixCSC{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(alpha,A,beta,switch2csr(B),indexA,indexB,indexC)
        end
        function geam(alpha::$elty,
                      A::CudaSparseMatrixCSC{$elty},
                      beta::$elty,
                      B::CudaSparseMatrixCSR{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(alpha,switch2csr(A),beta,B,indexA,indexB,indexC)
        end
        function geam(alpha::$elty,
                      A::Union{CudaSparseMatrixCSR{$elty},CudaSparseMatrixCSC{$elty}},
                      B::Union{CudaSparseMatrixCSR{$elty},CudaSparseMatrixCSC{$elty}},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(alpha,A,one($elty),B,indexA,indexB,indexC)
        end
        function geam(A::Union{CudaSparseMatrixCSR{$elty},CudaSparseMatrixCSC{$elty}},
                      beta::$elty,
                      B::Union{CudaSparseMatrixCSR{$elty},CudaSparseMatrixCSC{$elty}},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(one($elty),A,beta,B,indexA,indexB,indexC)
        end
        function geam(A::Union{CudaSparseMatrixCSR{$elty},CudaSparseMatrixCSC{$elty}},
                      B::Union{CudaSparseMatrixCSR{$elty},CudaSparseMatrixCSC{$elty}},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(one($elty),A,one($elty),B,indexA,indexB,indexC)
        end
    end
end

(+)(A::Union{CudaSparseMatrixCSR,CudaSparseMatrixCSC},B::Union{CudaSparseMatrixCSR,CudaSparseMatrixCSC}) = geam(A,B,'O','O','O')
(-)(A::Union{CudaSparseMatrixCSR,CudaSparseMatrixCSC},B::Union{CudaSparseMatrixCSR,CudaSparseMatrixCSC}) = geam(A,-one(eltype(A)),B,'O','O','O')

#CSR GEMM
for (fname,elty) in ((:cusparseScsrgemm, :Float32),
                     (:cusparseDcsrgemm, :Float64),
                     (:cusparseCcsrgemm, :Complex64),
                     (:cusparseZcsrgemm, :Complex128))
    @eval begin
        function gemm(transa::SparseChar,
                      transb::SparseChar,
                      A::CudaSparseMatrixCSR{$elty},
                      B::CudaSparseMatrixCSR{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            cutransa = cusparseop(transa)
            cutransb = cusparseop(transb)
            cuinda   = cusparseindex(indexA)
            cuindb   = cusparseindex(indexB)
            cuindc   = cusparseindex(indexB)
            cudesca  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescb  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindb)
            cudescc  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            m,k  = transa == 'N' ? A.dims : (A.dims[2],A.dims[1])
            kB,n = transb == 'N' ? B.dims : (B.dims[2],B.dims[1])
            if k != kB
                throw(DimensionMismatch("Interior dimension of A, $k, and B, $kB, must match"))
            end
            nnzC = Array(Cint,1)
            rowPtrC = CudaArray(zeros(Cint,m + 1))
            statuscheck(ccall((:cusparseXcsrgemmNnz,libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint},
                               Ptr{Cint}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint},
                               Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{Cint},
                               Ptr{Cint}), cusparsehandle[1], cutransa, cutransb,
                               m, n, k, &cudesca, A.nnz, A.rowPtr, A.colVal,
                               &cudescb, B.nnz, B.rowPtr, B.colVal, &cudescc,
                               rowPtrC, nnzC))
            nnz = nnzC[1]
            C = CudaSparseMatrixCSR(rowPtrC, CudaArray(zeros(Cint,nnz)), CudaArray(zeros($elty,nnz)), nnz, (m,n))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t},
                               Cint, Ptr{$elty}, Ptr{Cint}, Ptr{Cint},
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}), cusparsehandle[1], cutransa,
                               cutransb, m, n, k, &cudesca, A.nnz, A.nzVal,
                               A.rowPtr, A.colVal, &cudescb, B.nnz, B.nzVal,
                               B.rowPtr, B.colVal, &cudescc, C.nzVal,
                               C.rowPtr, C.colVal))
            C
        end
    end
end

#CSC GEMM
for (fname,elty) in ((:cusparseScsrgemm, :Float32),
                     (:cusparseDcsrgemm, :Float64),
                     (:cusparseCcsrgemm, :Complex64),
                     (:cusparseZcsrgemm, :Complex128))
    @eval begin
        function gemm(transa::SparseChar,
                      transb::SparseChar,
                      A::CudaSparseMatrixCSC{$elty},
                      B::CudaSparseMatrixCSC{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cutransa = cusparseop(ctransa)
            ctransb = 'N'
            if transb == 'N'
                ctransb = 'T'
            end
            cutransb = cusparseop(ctransb)
            cuinda   = cusparseindex(indexA)
            cuindb   = cusparseindex(indexB)
            cuindc   = cusparseindex(indexB)
            cudesca  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescb  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindb)
            cudescc  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            m,k  = ctransa != 'N' ? A.dims : (A.dims[2],A.dims[1])
            kB,n = ctransb != 'N' ? B.dims : (B.dims[2],B.dims[1])
            if k != kB
                throw(DimensionMismatch("Interior dimension of A, $k, and B, $kB, must match"))
            end
            nnzC = Array(Cint,1)
            colPtrC = CudaArray(zeros(Cint,n + 1))
            statuscheck(ccall((:cusparseXcsrgemmNnz,libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint},
                               Ptr{Cint}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint},
                               Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{Cint},
                               Ptr{Cint}), cusparsehandle[1], cutransa, cutransb,
                               m, n, k, &cudesca, A.nnz, A.colPtr, A.rowVal,
                               &cudescb, B.nnz, B.colPtr, B.rowVal, &cudescc,
                               colPtrC, nnzC))
            nnz = nnzC[1]
            C = CudaSparseMatrixCSC(colPtrC, CudaArray(zeros(Cint,nnz)), CudaArray(zeros($elty,nnz)), nnz, (m,n))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t},
                               Cint, Ptr{$elty}, Ptr{Cint}, Ptr{Cint},
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}), cusparsehandle[1], cutransa,
                               cutransb, m, n, k, &cudesca, A.nnz, A.nzVal,
                               A.colPtr, A.rowVal, &cudescb, B.nnz, B.nzVal,
                               B.colPtr, B.rowVal, &cudescc, C.nzVal,
                               C.colPtr, C.rowVal))
            C
        end
    end
end

## preconditioners

# ic0 - incomplete Cholesky factorization with no pivoting

for (fname,elty) in ((:cusparseScsric0, :Float32),
                     (:cusparseDcsric0, :Float64),
                     (:cusparseCcsric0, :Complex64),
                     (:cusparseZcsric0, :Complex128))
    @eval begin
        function ic0!(transa::SparseChar,
                      typea::SparseChar,
                      A::CompressedSparse{$elty},
                      info::cusparseSolveAnalysisInfo_t,
                      index::SparseChar)
            Mat     = A
            if typeof(A) <: Base.LinAlg.HermOrSym
                Mat = A.data
            end
            cutransa = cusparseop(transa)
            cutype   = cusparsetype(typea)
            if typeof(A) <: Symmetric
                cutype = cusparsetype('S')
            elseif typeof(A) <: Hermitian
                cutype = cusparsetype('H')
            end

            if transa == 'N' && typeof(Mat) == CudaSparseMatrixCSC{$elty}
                cutransa = cusparseop('T')
            elseif transa == 'T' && typeof(Mat) == CudaSparseMatrixCSC{$elty}
                cutransa = cusparseop('N')
            end
            cuind    = cusparseindex(index)
            cudesc   = getDescr(A, index)
            m,n      = Mat.dims
            indPtr   = typeof(Mat) == CudaSparseMatrixCSC{$elty} ? Mat.colPtr : Mat.rowPtr
            valPtr   = typeof(Mat) == CudaSparseMatrixCSC{$elty} ? Mat.rowVal : Mat.colVal
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, cusparseSolveAnalysisInfo_t),
                              cusparsehandle[1], cutransa, m, &cudesc, Mat.nzVal,
                              indPtr, valPtr, info))
            Mat
        end
        function ic0(transa::SparseChar,
                     typea::SparseChar,
                     A::CompressedSparse{$elty},
                     info::cusparseSolveAnalysisInfo_t,
                     index::SparseChar)
            ic0!(transa,typea,copy(A),info,index)
        end
    end
end

# csric02
for (bname,aname,sname,elty) in ((:cusparseScsric02_bufferSize, :cusparseScsric02_analysis, :cusparseScsric02, :Float32),
                                 (:cusparseDcsric02_bufferSize, :cusparseDcsric02_analysis, :cusparseDcsric02, :Float64),
                                 (:cusparseCcsric02_bufferSize, :cusparseCcsric02_analysis, :cusparseCcsric02, :Complex64),
                                 (:cusparseZcsric02_bufferSize, :cusparseZcsric02_analysis, :cusparseZcsric02, :Complex128))
    @eval begin
        function ic02!(A::CudaSparseMatrixCSR{$elty},
                       index::SparseChar)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csric02Info_t[0]
            cusparseCreateCsric02Info(info)
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csric02Info_t, Ptr{Cint}),
                              cusparsehandle[1], m, A.nnz, &cudesc, A.nzVal,
                              A.rowPtr, A.colVal, info[1], bufSize))
            buffer = CudaArray(zeros(UInt8, bufSize[1]))
            statuscheck(ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                               Ptr{Void}), cusparsehandle[1], m, A.nnz, &cudesc,
                               A.nzVal, A.rowPtr, A.colVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            posit = Array(Cint,1)
            statuscheck(ccall((:cusparseXcsric02_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, csric02Info_t,
                        Ptr{Cint}), cusparsehandle[1], info[1], posit))
            if( posit[1] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[1],posit[1],")"))
            end
            statuscheck(ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                               Ptr{Void}), cusparsehandle[1], m, A.nnz,
                               &cudesc, A.nzVal, A.rowPtr, A.colVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            cusparseDestroyCsric02Info(info[1])
            A
        end
    end
end

# cscic02
for (bname,aname,sname,elty) in ((:cusparseScsric02_bufferSize, :cusparseScsric02_analysis, :cusparseScsric02, :Float32),
                                 (:cusparseDcsric02_bufferSize, :cusparseDcsric02_analysis, :cusparseDcsric02, :Float64),
                                 (:cusparseCcsric02_bufferSize, :cusparseCcsric02_analysis, :cusparseCcsric02, :Complex64),
                                 (:cusparseZcsric02_bufferSize, :cusparseZcsric02_analysis, :cusparseZcsric02, :Complex128))
    @eval begin
        function ic02!(A::CudaSparseMatrixCSC{$elty},
                       index::SparseChar)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csric02Info_t[0]
            cusparseCreateCsric02Info(info)
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csric02Info_t, Ptr{Cint}),
                              cusparsehandle[1], m, A.nnz, &cudesc, A.nzVal,
                              A.colPtr, A.rowVal, info[1], bufSize))
            buffer = CudaArray(zeros(UInt8, bufSize[1]))
            statuscheck(ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                               Ptr{Void}), cusparsehandle[1], m, A.nnz, &cudesc,
                               A.nzVal, A.colPtr, A.rowVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            posit = Array(Cint,1)
            statuscheck(ccall((:cusparseXcsric02_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, csric02Info_t,
                        Ptr{Cint}), cusparsehandle[1], info[1], posit))
            if( posit[1] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[1],posit[1],")"))
            end
            statuscheck(ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                               Ptr{Void}), cusparsehandle[1], m, A.nnz,
                               &cudesc, A.nzVal, A.colPtr, A.rowVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            cusparseDestroyCsric02Info(info[1])
            A
        end
    end
end

# ilu0 - incomplete LU factorization with no pivoting

for (fname,elty) in ((:cusparseScsrilu0, :Float32),
                     (:cusparseDcsrilu0, :Float64),
                     (:cusparseCcsrilu0, :Complex64),
                     (:cusparseZcsrilu0, :Complex128))
    @eval begin
        function ilu0!(transa::SparseChar,
                       A::CompressedSparse{$elty},
                       info::cusparseSolveAnalysisInfo_t,
                       index::SparseChar)
            Mat = A
            if typeof(A) <: Base.LinAlg.HermOrSym
                Mat = A.data
            end
            cutransa = cusparseop(transa)
            if transa == 'N' && typeof(Mat) == CudaSparseMatrixCSC{$elty}
                cutransa = cusparseop('T')
            elseif transa == 'T' && typeof(Mat) == CudaSparseMatrixCSC{$elty}
                cutransa = cusparseop('N')
            end
            cuind    = cusparseindex(index)
            cudesc   = getDescr(A, index)
            m,n      = Mat.dims
            indPtr   = typeof(Mat) == CudaSparseMatrixCSC{$elty} ? Mat.colPtr : Mat.rowPtr
            valPtr   = typeof(Mat) == CudaSparseMatrixCSC{$elty} ? Mat.rowVal : Mat.colVal
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, cusparseSolveAnalysisInfo_t),
                              cusparsehandle[1], cutransa, m, &cudesc, Mat.nzVal,
                              indPtr, valPtr, info))
            Mat
        end
        function ilu0(transa::SparseChar,
                      A::CompressedSparse{$elty},
                      info::cusparseSolveAnalysisInfo_t,
                      index::SparseChar)
            ilu0!(transa,copy(A),info,index)
        end
    end
end

# csrilu02
for (bname,aname,sname,elty) in ((:cusparseScsrilu02_bufferSize, :cusparseScsrilu02_analysis, :cusparseScsrilu02, :Float32),
                                 (:cusparseDcsrilu02_bufferSize, :cusparseDcsrilu02_analysis, :cusparseDcsrilu02, :Float64),
                                 (:cusparseCcsrilu02_bufferSize, :cusparseCcsrilu02_analysis, :cusparseCcsrilu02, :Complex64),
                                 (:cusparseZcsrilu02_bufferSize, :cusparseZcsrilu02_analysis, :cusparseZcsrilu02, :Complex128))
    @eval begin
        function ilu02!(A::CudaSparseMatrixCSR{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csrilu02Info_t[0]
            cusparseCreateCsrilu02Info(info)
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csrilu02Info_t, Ptr{Cint}),
                              cusparsehandle[1], m, A.nnz, &cudesc, A.nzVal,
                              A.rowPtr, A.colVal, info[1], bufSize))
            buffer = CudaArray(zeros(UInt8, bufSize[1]))
            statuscheck(ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                               Ptr{Void}), cusparsehandle[1], m, A.nnz, &cudesc,
                               A.nzVal, A.rowPtr, A.colVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            posit = Array(Cint,1)
            statuscheck(ccall((:cusparseXcsrilu02_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, csrilu02Info_t,
                        Ptr{Cint}), cusparsehandle[1], info[1], posit))
            if( posit[1] >= 0 )
                throw(string("Structural zero in A at (",posit[1],posit[1],")"))
            end
            statuscheck(ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                               Ptr{Void}), cusparsehandle[1], m, A.nnz,
                               &cudesc, A.nzVal, A.rowPtr, A.colVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            cusparseDestroyCsrilu02Info(info[1])
            A
        end
    end
end

# cscilu02
for (bname,aname,sname,elty) in ((:cusparseScsrilu02_bufferSize, :cusparseScsrilu02_analysis, :cusparseScsrilu02, :Float32),
                                 (:cusparseDcsrilu02_bufferSize, :cusparseDcsrilu02_analysis, :cusparseDcsrilu02, :Float64),
                                 (:cusparseCcsrilu02_bufferSize, :cusparseCcsrilu02_analysis, :cusparseCcsrilu02, :Complex64),
                                 (:cusparseZcsrilu02_bufferSize, :cusparseZcsrilu02_analysis, :cusparseZcsrilu02, :Complex128))
    @eval begin
        function ilu02!(A::CudaSparseMatrixCSC{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csrilu02Info_t[0]
            cusparseCreateCsrilu02Info(info)
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csrilu02Info_t, Ptr{Cint}),
                              cusparsehandle[1], m, A.nnz, &cudesc, A.nzVal,
                              A.colPtr, A.rowVal, info[1], bufSize))
            buffer = CudaArray(zeros(UInt8, bufSize[1]))
            statuscheck(ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                               Ptr{Void}), cusparsehandle[1], m, A.nnz, &cudesc,
                               A.nzVal, A.colPtr, A.rowVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            posit = Array(Cint,1)
            statuscheck(ccall((:cusparseXcsrilu02_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, csrilu02Info_t,
                        Ptr{Cint}), cusparsehandle[1], info[1], posit))
            if( posit[1] >= 0 )
                throw(string("Structural zero in A at (",posit[1],posit[1],")"))
            end
            statuscheck(ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                               Ptr{Void}), cusparsehandle[1], m, A.nnz,
                               &cudesc, A.nzVal, A.colPtr, A.rowVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            cusparseDestroyCsrilu02Info(info[1])
            A
        end
    end
end

# bsric02
for (bname,aname,sname,elty) in ((:cusparseSbsric02_bufferSize, :cusparseSbsric02_analysis, :cusparseSbsric02, :Float32),
                                 (:cusparseDbsric02_bufferSize, :cusparseDbsric02_analysis, :cusparseDbsric02, :Float64),
                                 (:cusparseCbsric02_bufferSize, :cusparseCbsric02_analysis, :cusparseCbsric02, :Complex64),
                                 (:cusparseZbsric02_bufferSize, :cusparseZbsric02_analysis, :cusparseZbsric02, :Complex128))
    @eval begin
        function ic02!(A::CudaSparseMatrixBSR{$elty},
                       index::SparseChar)
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            info = bsric02Info_t[0]
            cusparseCreateBsric02Info(info)
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Cint, bsric02Info_t,
                               Ptr{Cint}), cusparsehandle[1], cudir, mb, A.nnz,
                               &cudesc, A.nzVal, A.rowPtr, A.colVal,
                               A.blockDim, info[1], bufSize))
            buffer = CudaArray(zeros(UInt8, bufSize[1]))
            statuscheck(ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Cint, bsric02Info_t,
                               cusparseSolvePolicy_t, Ptr{Void}),
                              cusparsehandle[1], cudir, mb, A.nnz, &cudesc,
                              A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            posit = Array(Cint,1)
            statuscheck(ccall((:cusparseXbsric02_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, bsric02Info_t,
                        Ptr{Cint}), cusparsehandle[1], info[1], posit))
            if( posit[1] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[1],posit[1],")"))
            end
            statuscheck(ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Cint,bsric02Info_t,
                               cusparseSolvePolicy_t, Ptr{Void}),
                              cusparsehandle[1], cudir, mb, A.nnz, &cudesc,
                              A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            cusparseDestroyBsric02Info(info[1])
            A
        end
    end
end

# bsrilu02
for (bname,aname,sname,elty) in ((:cusparseSbsrilu02_bufferSize, :cusparseSbsrilu02_analysis, :cusparseSbsrilu02, :Float32),
                                 (:cusparseDbsrilu02_bufferSize, :cusparseDbsrilu02_analysis, :cusparseDbsrilu02, :Float64),
                                 (:cusparseCbsrilu02_bufferSize, :cusparseCbsrilu02_analysis, :cusparseCbsrilu02, :Complex64),
                                 (:cusparseZbsrilu02_bufferSize, :cusparseZbsrilu02_analysis, :cusparseZbsrilu02, :Complex128))
    @eval begin
        function ilu02!(A::CudaSparseMatrixBSR{$elty},
                        index::SparseChar)
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            info = bsrilu02Info_t[0]
            cusparseCreateBsrilu02Info(info)
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Cint, bsrilu02Info_t,
                               Ptr{Cint}), cusparsehandle[1], cudir, mb, A.nnz,
                               &cudesc, A.nzVal, A.rowPtr, A.colVal,
                               A.blockDim, info[1], bufSize))
            buffer = CudaArray(zeros(UInt8, bufSize[1]))
            statuscheck(ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Cint, bsrilu02Info_t,
                               cusparseSolvePolicy_t, Ptr{Void}),
                              cusparsehandle[1], cudir, mb, A.nnz, &cudesc,
                              A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            posit = Array(Cint,1)
            statuscheck(ccall((:cusparseXbsrilu02_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, bsrilu02Info_t,
                        Ptr{Cint}), cusparsehandle[1], info[1], posit))
            if( posit[1] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[1],posit[1],")"))
            end
            statuscheck(ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Cint,bsrilu02Info_t,
                               cusparseSolvePolicy_t, Ptr{Void}),
                              cusparsehandle[1], cudir, mb, A.nnz, &cudesc,
                              A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer))
            cusparseDestroyBsrilu02Info(info[1])
            A
        end
    end
end

for elty in (:Float32, :Float64, :Complex64, :Complex128)
    @eval begin
        function ilu02(A::CudaSparseMatrix{$elty},
                       index::SparseChar)
            ilu02!(copy(A),index)
        end
        function ic02(A::CudaSparseMatrix{$elty},
                      index::SparseChar)
            ic02!(copy(A),index)
        end
        function ilu02(A::HermOrSym{$elty,CudaSparseMatrix{$elty}},
                       index::SparseChar)
            ilu02!(copy(A.data),index)
        end
        function ic02(A::HermOrSym{$elty,CudaSparseMatrix{$elty}},
                      index::SparseChar)
            ic02!(copy(A.data),index)
        end
    end
end

# gtsv - general tridiagonal solver
for (fname,elty) in ((:cusparseSgtsv, :Float32),
                     (:cusparseDgtsv, :Float64),
                     (:cusparseCgtsv, :Complex64),
                     (:cusparseZgtsv, :Complex128))
    @eval begin
        function gtsv!(dl::CudaVector{$elty},
                       d::CudaVector{$elty},
                       du::CudaVector{$elty},
                       B::CudaMatrix{$elty})
            m,n = B.dims
            ldb = max(1,stride(B,2))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, Ptr{$elty},
                               Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Cint),
                              cusparsehandle[1], m, n, dl, d, du, B, ldb))
            B
        end
        function gtsv(dl::CudaVector{$elty},
                      d::CudaVector{$elty},
                      du::CudaVector{$elty},
                      B::CudaMatrix{$elty})
            gtsv!(dl,d,du,copy(B))
        end
    end
end

# gtsv_nopivot - general tridiagonal solver without pivoting
for (fname,elty) in ((:cusparseSgtsv_nopivot, :Float32),
                     (:cusparseDgtsv_nopivot, :Float64),
                     (:cusparseCgtsv_nopivot, :Complex64),
                     (:cusparseZgtsv_nopivot, :Complex128))
    @eval begin
        function gtsv_nopivot!(dl::CudaVector{$elty},
                               d::CudaVector{$elty},
                               du::CudaVector{$elty},
                               B::CudaMatrix{$elty})
            m,n = B.dims
            ldb = max(1,stride(B,2))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, Ptr{$elty},
                               Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Cint),
                              cusparsehandle[1], m, n, dl, d, du, B, ldb))
            B
        end
        function gtsv_nopivot(dl::CudaVector{$elty},
                              d::CudaVector{$elty},
                              du::CudaVector{$elty},
                              B::CudaMatrix{$elty})
            gtsv_nopivot!(dl,d,du,copy(B))
        end
    end
end

# gtsvStridedBatch - batched general tridiagonal solver
for (fname,elty) in ((:cusparseSgtsvStridedBatch, :Float32),
                     (:cusparseDgtsvStridedBatch, :Float64),
                     (:cusparseCgtsvStridedBatch, :Complex64),
                     (:cusparseZgtsvStridedBatch, :Complex128))
    @eval begin
        function gtsvStridedBatch!(dl::CudaVector{$elty},
                                   d::CudaVector{$elty},
                                   du::CudaVector{$elty},
                                   X::CudaVector{$elty},
                                   batchCount::Integer,
                                   batchStride::Integer)
            m = div(length(X),batchCount)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{$elty},
                               Ptr{$elty}, Ptr{$elty}, Cint, Cint),
                              cusparsehandle[1], m, dl, d, du, X,
                              batchCount, batchStride))
            X
        end
        function gtsvStridedBatch(dl::CudaVector{$elty},
                                  d::CudaVector{$elty},
                                  du::CudaVector{$elty},
                                  X::CudaVector{$elty},
                                  batchCount::Integer,
                                  batchStride::Integer)
            gtsvStridedBatch!(dl,d,du,copy(X),batchCount,batchStride)
        end
    end
end
