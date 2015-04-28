#utilities

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
    throw("unknown cusparse operation.")
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
    throw("unknown cusparse matrix type.")
end

# convert SparseChar {U,L} to cusparseFillMode_t
function cusparsefill(uplo::SparseChar)
    if uplo == 'U'
        return CUSPARSE_FILL_MODE_UPPER
    end
    if uplo == 'L'
        return CUSPARSE_FILL_MODE_LOWER
    end
    throw("unknown cusparse fill mode")
end

# convert SparseChar {U,N} to cusparseDiagType_t
function cusparsediag(diag::SparseChar)
    if diag == 'U'
        return CUSPARSE_DIAG_UNIT
    end
    if diag == 'N'
        return CUSPARSE_DIAG_NON_UNIT
    end
    throw("unknown cusparse diag mode")
end

# convert SparseChar {Z,O} to cusparseIndexBase_t
function cusparseindex(index::SparseChar)
    if index == 'Z'
        return CUSPARSE_INDEX_BASE_ZERO
    end
    if index == 'O'
        return CUSPARSE_INDEX_BASE_ONE
    end
    throw("unknown cusparse index base")
end

# convert SparseChar {R,C} to cusparseDirection_t
function cusparsedir(dir::SparseChar)
    if dir == 'R'
        return CUSPARSE_DIRECTION_ROW
    end
    if dir == 'C'
        return CUSPARSE_DIRECTION_COL
    end
    throw("unknown cusparse direction")
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
            csc = CudaSparseMatrixCSC(eltype(csr),colPtr,rowVal,nzVal,csr.nnz,csr.dims)
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
            csr = CudaSparseMatrixCSR(eltype(csc),rowPtr,colVal,nzVal,csc.nnz,csc.dims)
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
            CudaSparseMatrixCSR($elty, csrRowPtr, csrColInd, csrNzVal, convert(Cint,nnz), bsr.dims)
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
                               Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint}),
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
                               Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint}),
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
                return CudaSparseMatrixCSR($elty,rowPtr,colInd,nzVal,nnzTotal[1],size(A))
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
                return CudaSparseMatrixCSC($elty,colPtr,rowInd,nzVal,nnzTotal[1],size(A))
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
            CudaSparseMatrixCSR($elty, csrRowPtr, csrColInd, csrNzVal, hyb.nnz, hyb.dims)
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
                        X::CudaSparseMatrixCSC{$elty},
                        Y::CudaVector{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{$elty},
                               Ptr{Cint}, Ptr{$elty}, cusparseIndexBase_t),
                              cusparsehandle[1], X.nnz, [alpha], X.nzVal, X.rowVal,
                              Y, cuind))
            Y
        end
        function axpyi(alpha::$elty,
                       X::CudaSparseMatrixCSC{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            axpyi!(alpha,X,copy(Y),index)
        end
        function axpyi(X::CudaSparseMatrixCSC{$elty},
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
        function $jname(X::CudaSparseMatrixCSC{$elty},
                        Y::CudaVector{$elty},
                        index::SparseChar)
            dot = Array($elty,1)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{Cint},
                               Ptr{$elty}, Ptr{$elty}, cusparseIndexBase_t),
                              cusparsehandle[1], X.nnz, X.nzVal, X.rowVal,
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
        function gthr!(X::CudaSparseMatrixCSC{$elty},
                      Y::CudaVector{$elty},
                      index::SparseChar)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{$elty},
                               Ptr{Cint}, cusparseIndexBase_t), cusparsehandle[1],
                              X.nnz, Y, X.nzVal, X.rowVal, cuind))
            X
        end
        function gthr(X::CudaSparseMatrixCSC{$elty},
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
        function gthrz!(X::CudaSparseMatrixCSC{$elty},
                        Y::CudaVector{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{$elty},
                               Ptr{Cint}, cusparseIndexBase_t), cusparsehandle[1],
                              X.nnz, Y, X.nzVal, X.rowVal, cuind))
            X,Y
        end
        function gthrz(X::CudaSparseMatrixCSC{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            gthrz!(copy(X),copy(Y),index)
        end
    end
end

for (fname,elty) in ((:cusparseSroti, :Float32),
                     (:cusparseDroti, :Float64))
    @eval begin
        function roti!(X::CudaSparseMatrixCSC{$elty},
                       Y::CudaVector{$elty},
                       c::$elty,
                       s::$elty,
                       index::SparseChar)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{$Cint},
                               Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, cusparseIndexBase_t),
                              cusparsehandle[1], X.nnz, X.nzVal, X.rowVal, Y, [c], [s], cuind))
            X,Y
        end
        function roti(X::CudaSparseMatrixCSC{$elty},
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
        function sctr!(X::CudaSparseMatrixCSC{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{Cint},
                               Ptr{$elty}, cusparseIndexBase_t),
                              cusparsehandle[1], X.nnz, X.nzVal, X.rowVal,
                              Y, cuind))
            Y
        end
        function sctr(X::CudaSparseMatrixCSC{$elty},
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
        function bsrmv!(transa::SparseChar,
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
            if( transa == 'N' && (length(X) != n || length(Y) != m) )
                throw(DimensionMismatch(""))
            end
            if( (transa == 'T' || transa == 'C') && (length(X) != m || length(Y) != n) )
                throw(DimensionMismatch(""))
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
        function bsrmv(transa::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixBSR{$elty},
                       X::CudaVector{$elty},
                       beta::$elty,
                       Y::CudaVector{$elty},
                       index::SparseChar)
            bsrmv!(transa,alpha,A,X,beta,copy(Y),index)
        end
        function bsrmv(transa::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixBSR{$elty},
                       X::CudaVector{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            bsrmv(transa,alpha,A,X,one($elty),Y,index)
        end
        function bsrmv(transa::SparseChar,
                       A::CudaSparseMatrixBSR{$elty},
                       X::CudaVector{$elty},
                       beta::$elty,
                       Y::CudaVector{$elty},
                       index::SparseChar)
            bsrmv(transa,one($elty),A,X,beta,Y,index)
        end
        function bsrmv(transa::SparseChar,
                       A::CudaSparseMatrixBSR{$elty},
                       X::CudaVector{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            bsrmv(transa,one($elty),A,X,one($elty),Y,index)
        end
        function bsrmv(transa::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixBSR{$elty},
                       X::CudaVector{$elty},
                       index::SparseChar)
            bsrmv(transa,alpha,A,X,zero($elty),CudaArray(zeros($elty,size(A)[1])),index)
        end
        function bsrmv(transa::SparseChar,
                       A::CudaSparseMatrixBSR{$elty},
                       X::CudaVector{$elty},
                       index::SparseChar)
            bsrmv(transa,one($elty),A,X,zero($elty),CudaArray(zeros($elty,size(A)[1])),index)
        end
    end
end

for (fname,elty) in ((:cusparseScsrmv, :Float32),
                     (:cusparseDcsrmv, :Float64),
                     (:cusparseCcsrmv, :Complex64),
                     (:cusparseZcsrmv, :Complex128))
    @eval begin
        function csrmv!(transa::SparseChar,
                        alpha::$elty,
                        A::CudaSparseMatrixCSR{$elty},
                        X::CudaVector{$elty},
                        beta::$elty,
                        Y::CudaVector{$elty},
                        index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( transa == 'N' && (length(X) != n || length(Y) != m) )
                throw(DimensionMismatch(""))
            end
            if( (transa == 'T' || transa == 'C') && (length(X) != m || length(Y) != n) )
                throw(DimensionMismatch(""))
            end
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                               Ptr{$elty}, Ptr{$elty}), cusparsehandle[1],
                               cutransa, m, n, A.nnz, [alpha], &cudesc, A.nzVal,
                               A.rowPtr, A.colVal, X, [beta], Y))
            Y
        end
        function csrmv(transa::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixCSR{$elty},
                       X::CudaVector{$elty},
                       beta::$elty,
                       Y::CudaVector{$elty},
                       index::SparseChar)
            csrmv!(transa,alpha,A,X,beta,copy(Y),index)
        end
        function csrmv(transa::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixCSR{$elty},
                       X::CudaVector{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            csrmv(transa,alpha,A,X,one($elty),Y,index)
        end
        function csrmv(transa::SparseChar,
                       A::CudaSparseMatrixCSR{$elty},
                       X::CudaVector{$elty},
                       beta::$elty,
                       Y::CudaVector{$elty},
                       index::SparseChar)
            csrmv(transa,one($elty),A,X,beta,Y,index)
        end
        function csrmv(transa::SparseChar,
                       A::CudaSparseMatrixCSR{$elty},
                       X::CudaVector{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            csrmv(transa,one($elty),A,X,one($elty),Y,index)
        end
        function csrmv(transa::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixCSR{$elty},
                       X::CudaVector{$elty},
                       index::SparseChar)
            csrmv(transa,alpha,A,X,zero($elty),CudaArray(zeros($elty,size(A)[1])),index)
        end
        function csrmv(transa::SparseChar,
                       A::CudaSparseMatrixCSR{$elty},
                       X::CudaVector{$elty},
                       index::SparseChar)
            csrmv(transa,one($elty),A,X,zero($elty),CudaArray(zeros($elty,size(A)[1])),index)
        end
    end
end

# bsrsv2
for (bname,aname,sname,elty) in ((:cusparseSbsrsv2_bufferSize, :cusparseSbsrsv2_analysis, :cusparseSbsrsv2_solve, :Float32),
                                 (:cusparseDbsrsv2_bufferSize, :cusparseDbsrsv2_analysis, :cusparseDbsrsv2_solve, :Float64),
                                 (:cusparseCbsrsv2_bufferSize, :cusparseCbsrsv2_analysis, :cusparseCbsrsv2_solve, :Complex64),
                                 (:cusparseZbsrsv2_bufferSize, :cusparseZbsrsv2_analysis, :cusparseZbsrsv2_solve, :Complex128))
    @eval begin
        function bsrsv2!(transa::SparseChar,
                         alpha::$elty,
                         A::CudaSparseMatrixBSR{$elty},
                         X::CudaVector{$elty},
                         index::SparseChar)
            cutransa  = cusparseop(transa)
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square!"))
            end
            mb = div(m,A.blockDim)
            mX = length(X)
            if( mX != m )
                throw(DimensionMismatch(""))
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
            buffer = CudaArray(zeros(Uint8, bufSize[1]))
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
        function bsrsv2(transa::SparseChar,
                        alpha::$elty,
                        A::CudaSparseMatrixBSR{$elty},
                        X::CudaVector{$elty},
                        index::SparseChar)
            bsrsv2!(transa,alpha,A,copy(X),index)
        end
    end
end

for (fname,elty) in ((:cusparseScsrsv_analysis, :Float32),
                     (:cusparseDcsrsv_analysis, :Float64),
                     (:cusparseCcsrsv_analysis, :Complex64),
                     (:cusparseZcsrsv_analysis, :Complex128))
    @eval begin
        function csrsv_analysis(transa::SparseChar,
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
                throw(DimensionMismatch("A must be square!"))
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

for (fname,elty) in ((:cusparseScsrsv_solve, :Float32),
                     (:cusparseDcsrsv_solve, :Float64),
                     (:cusparseCcsrsv_solve, :Complex64),
                     (:cusparseZcsrsv_solve, :Complex128))
    @eval begin
        function csrsv_solve(transa::SparseChar,
                             uplo::SparseChar,
                             alpha::$elty,
                             A::CudaSparseMatrixCSR{$elty},
                             X::CudaVector{$elty},
                             info::cusparseSolveAnalysisInfo_t,
                             index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( size(X)[1] != m )
                throw(DimensionMismatch(""))
            end
            Y = similar(X)
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

for (fname,elty) in ((:cusparseShybmv, :Float32),
                     (:cusparseDhybmv, :Float64),
                     (:cusparseChybmv, :Complex64),
                     (:cusparseZhybmv, :Complex128))
    @eval begin
        function hybmv!(transa::SparseChar,
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
            if( transa == 'N' && (length(X) != n || length(Y) != m) )
                throw(DimensionMismatch(""))
            end
            if( (transa == 'T' || transa == 'C') && (length(X) != m || length(Y) != n) )
                throw(DimensionMismatch(""))
            end
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               cusparseHybMat_t, Ptr{$elty},
                               Ptr{$elty}, Ptr{$elty}), cusparsehandle[1],
                               cutransa, [alpha], &cudesc, A.Mat, X, [beta], Y))
            Y
        end
        function hybmv(transa::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixHYB{$elty},
                       X::CudaVector{$elty},
                       beta::$elty,
                       Y::CudaVector{$elty},
                       index::SparseChar)
            hybmv!(transa,alpha,A,X,beta,copy(Y),index)
        end
        function hybmv(transa::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixHYB{$elty},
                       X::CudaVector{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            hybmv(transa,alpha,A,X,one($elty),Y,index)
        end
        function hybmv(transa::SparseChar,
                       A::CudaSparseMatrixHYB{$elty},
                       X::CudaVector{$elty},
                       beta::$elty,
                       Y::CudaVector{$elty},
                       index::SparseChar)
            hybmv(transa,one($elty),A,X,beta,Y,index)
        end
        function hybmv(transa::SparseChar,
                       A::CudaSparseMatrixHYB{$elty},
                       X::CudaVector{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            hybmv(transa,one($elty),A,X,one($elty),Y,index)
        end
        function hybmv(transa::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixHYB{$elty},
                       X::CudaVector{$elty},
                       index::SparseChar)
            hybmv(transa,alpha,A,X,zero($elty),CudaArray(zeros($elty,size(A)[1])),index)
        end
        function hybmv(transa::SparseChar,
                       A::CudaSparseMatrixHYB{$elty},
                       X::CudaVector{$elty},
                       index::SparseChar)
            hybmv(transa,one($elty),A,X,zero($elty),CudaArray(zeros($elty,size(A)[1])),index)
        end
    end
end

for (fname,elty) in ((:cusparseShybsv_analysis, :Float32),
                     (:cusparseDhybsv_analysis, :Float64),
                     (:cusparseChybsv_analysis, :Complex64),
                     (:cusparseZhybsv_analysis, :Complex128))
    @eval begin
        function hybsv_analysis(transa::SparseChar,
                                uplo::SparseChar,
                                A::CudaSparseMatrixHYB{$elty},
                                index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square!"))
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
        function hybsv_solve(transa::SparseChar,
                             uplo::SparseChar,
                             alpha::$elty,
                             A::CudaSparseMatrixHYB{$elty},
                             X::CudaVector{$elty},
                             info::cusparseSolveAnalysisInfo_t,
                             index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( size(X)[1] != m )
                throw(DimensionMismatch(""))
            end
            Y = similar(X)
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

## level 3 functions

# bsrmm
for (fname,elty) in ((:cusparseSbsrmm, :Float32),
                     (:cusparseDbsrmm, :Float64),
                     (:cusparseCbsrmm, :Complex64),
                     (:cusparseZbsrmm, :Complex128))
    @eval begin
        function bsrmm!(transa::SparseChar,
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
            if( transa == 'N' && ( (transb == 'N' ? size(B) != (k,n) : size(B) != (n,k)) || size(C) != (m,n)) )
                throw(DimensionMismatch(""))
            end
            if( (transa == 'T' || transa == 'C') && ((transb == 'N' ? size(B) != (m,n) : size(B) != (n,m)) || size(C) != (k,n)) )
                throw(DimensionMismatch(""))
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
        function bsrmm(transa::SparseChar,
                       transb::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixBSR{$elty},
                       B::CudaMatrix{$elty},
                       beta::$elty,
                       C::CudaMatrix{$elty},
                       index::SparseChar)
            bsrmm!(transa,transb,alpha,A,B,beta,copy(C),index)
        end
        function bsrmm(transa::SparseChar,
                       transb::SparseChar,
                       A::CudaSparseMatrixBSR{$elty},
                       B::CudaMatrix{$elty},
                       beta::$elty,
                       C::CudaMatrix{$elty},
                       index::SparseChar)
            bsrmm(transa,transb,one($elty),A,B,beta,C,index)
        end
        function bsrmm(transa::SparseChar,
                       transb::SparseChar,
                       A::CudaSparseMatrixBSR{$elty},
                       B::CudaMatrix{$elty},
                       C::CudaMatrix{$elty},
                       index::SparseChar)
            bsrmm(transa,transb,one($elty),A,B,one($elty),C,index)
        end
        function bsrmm(transa::SparseChar,
                       transb::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixBSR{$elty},
                       B::CudaMatrix{$elty},
                       index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            bsrmm!(transa,transb,alpha,A,B,zero($elty),CudaArray(zeros($elty,(m,size(B)[2]))),index)
        end
        function bsrmm(transa::SparseChar,
                       transb::SparseChar,
                       A::CudaSparseMatrixBSR{$elty},
                       B::CudaMatrix{$elty},
                       index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            bsrmm!(transa,transb,one($elty),A,B,zero($elty),CudaArray(zeros($elty,(m,size(B)[2]))),index)
        end
    end
end

# csrmm
for (fname,elty) in ((:cusparseScsrmm, :Float32),
                     (:cusparseDcsrmm, :Float64),
                     (:cusparseCcsrmm, :Complex64),
                     (:cusparseZcsrmm, :Complex128))
    @eval begin
        function csrmm!(transa::SparseChar,
                        alpha::$elty,
                        A::CudaSparseMatrixCSR{$elty},
                        B::CudaMatrix{$elty},
                        beta::$elty,
                        C::CudaMatrix{$elty},
                        index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,k = A.dims
            n = size(C)[2]
            if( transa == 'N' && (size(B) != (k,n) || size(C) != (m,n)) )
                throw(DimensionMismatch(""))
            end
            if( (transa == 'T' || transa == 'C') && (size(B) != (m,n) || size(C) != (k,n)) )
                throw(DimensionMismatch(""))
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                               Cint, Ptr{$elty}, Ptr{$elty}, Cint),
                               cusparsehandle[1], cutransa, m, n, k, A.nnz,
                               [alpha], &cudesc, A.nzVal,A.rowPtr, A.colVal, B,
                               ldb, [beta], C, ldc))
            C
        end
        function csrmm(transa::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixCSR{$elty},
                       B::CudaMatrix{$elty},
                       beta::$elty,
                       C::CudaMatrix{$elty},
                       index::SparseChar)
            csrmm!(transa,alpha,A,B,beta,copy(C),index)
        end
        function csrmm(transa::SparseChar,
                       A::CudaSparseMatrixCSR{$elty},
                       B::CudaMatrix{$elty},
                       beta::$elty,
                       C::CudaMatrix{$elty},
                       index::SparseChar)
            csrmm(transa,one($elty),A,B,beta,C,index)
        end
        function csrmm(transa::SparseChar,
                       A::CudaSparseMatrixCSR{$elty},
                       B::CudaMatrix{$elty},
                       C::CudaMatrix{$elty},
                       index::SparseChar)
            csrmm(transa,one($elty),A,B,one($elty),C,index)
        end
        function csrmm(transa::SparseChar,
                       alpha::$elty,
                       A::CudaSparseMatrixCSR{$elty},
                       B::CudaMatrix{$elty},
                       index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            csrmm!(transa,alpha,A,B,zero($elty),CudaArray(zeros($elty,(m,size(B)[2]))),index)
        end
        function csrmm(transa::SparseChar,
                       A::CudaSparseMatrixCSR{$elty},
                       B::CudaMatrix{$elty},
                       index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            csrmm!(transa,one($elty),A,B,zero($elty),CudaArray(zeros($elty,(m,size(B)[2]))),index)
        end
    end
end

for (fname,elty) in ((:cusparseScsrmm2, :Float32),
                     (:cusparseDcsrmm2, :Float64),
                     (:cusparseCcsrmm2, :Complex64),
                     (:cusparseZcsrmm2, :Complex128))
    @eval begin
        function csrmm2!(transa::SparseChar,
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
            if( transa == 'N' && ( (transb == 'N' ? size(B) != (k,n) : size(B) != (n,k)) || size(C) != (m,n)) )
                throw(DimensionMismatch(""))
            end
            if( (transa == 'T' || transa == 'C') && ((transb == 'N' ? size(B) != (m,n) : size(B) != (n,m)) || size(C) != (k,n)) )
                throw(DimensionMismatch(""))
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
        function csrmm2(transa::SparseChar,
                        transb::SparseChar,
                        alpha::$elty,
                        A::CudaSparseMatrixCSR{$elty},
                        B::CudaMatrix{$elty},
                        beta::$elty,
                        C::CudaMatrix{$elty},
                        index::SparseChar)
            csrmm2!(transa,transb,alpha,A,B,beta,copy(C),index)
        end
        function csrmm2(transa::SparseChar,
                        transb::SparseChar,
                        A::CudaSparseMatrixCSR{$elty},
                        B::CudaMatrix{$elty},
                        beta::$elty,
                        C::CudaMatrix{$elty},
                        index::SparseChar)
            csrmm2(transa,transb,one($elty),A,B,beta,C,index)
        end
        function csrmm2(transa::SparseChar,
                        transb::SparseChar,
                        A::CudaSparseMatrixCSR{$elty},
                        B::CudaMatrix{$elty},
                        C::CudaMatrix{$elty},
                        index::SparseChar)
            csrmm2(transa,transb,one($elty),A,B,one($elty),C,index)
        end
        function csrmm2(transa::SparseChar,
                        transb::SparseChar,
                        alpha::$elty,
                        A::CudaSparseMatrixCSR{$elty},
                        B::CudaMatrix{$elty},
                        index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            n = transb == 'N' ? size(B)[2] : size(B)[1]
            csrmm2(transa,transb,alpha,A,B,zero($elty),CudaArray(zeros($elty,(m,n))),index)
        end
        function csrmm2(transa::SparseChar,
                        transb::SparseChar,
                        A::CudaSparseMatrixCSR{$elty},
                        B::CudaMatrix{$elty},
                        index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            n = transb == 'N' ? size(B)[2] : size(B)[1]
            csrmm2(transa,transb,one($elty),A,B,zero($elty),CudaArray(zeros($elty,(m,n))),index)
        end
    end
end

for (fname,elty) in ((:cusparseScsrsm_analysis, :Float32),
                     (:cusparseDcsrsm_analysis, :Float64),
                     (:cusparseCcsrsm_analysis, :Complex64),
                     (:cusparseZcsrsm_analysis, :Complex128))
    @eval begin
        function csrsm_analysis(transa::SparseChar,
                                uplo::SparseChar,
                                A::CudaSparseMatrixCSR{$elty},
                                index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( n != m )
                throw(DimensionMismatch("A must be square!"))
            end
            info = cusparseSolveAnalysisInfo_t[0]
            cusparseCreateSolveAnalysisInfo(info)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, cusparseSolveAnalysisInfo_t),
                              cusparsehandle[1], cutransa, m, A.nnz, &cudesc,
                              A.nzVal, A.rowPtr, A.colVal, info[1]))
            info[1]
        end
    end
end

for (fname,elty) in ((:cusparseScsrsm_solve, :Float32),
                     (:cusparseDcsrsm_solve, :Float64),
                     (:cusparseCcsrsm_solve, :Complex64),
                     (:cusparseZcsrsm_solve, :Complex128))
    @eval begin
        function csrsm_solve(transa::SparseChar,
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
                throw(DimensionMismatch(""))
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
    end
end

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
            buffer = CudaArray(zeros(Uint8, bufSize[1]))
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
            C = CudaSparseMatrixCSR($elty, rowPtrC, CudaArray(zeros(Cint,nnz)), CudaArray(zeros($elty,nnz)), nnz, A.dims)
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
                      A::CudaSparseMatrixCSR{$elty},
                      B::CudaSparseMatrixCSR{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(alpha,A,one($elty),B,indexA,indexB,indexC)
        end
        function geam(A::CudaSparseMatrixCSR{$elty},
                      beta::$elty,
                      B::CudaSparseMatrixCSR{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(one($elty),A,beta,B,indexA,indexB,indexC)
        end
        function geam(A::CudaSparseMatrixCSR{$elty},
                      B::CudaSparseMatrixCSR{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(one($elty),A,one($elty),B,indexA,indexB,indexC)
        end
    end
end

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
            cutransa = cusparseop(transb)
            cutransb = cusparseop(transa)
            cuinda = cusparseindex(indexA)
            cuindb = cusparseindex(indexB)
            cuindc = cusparseindex(indexB)
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescb = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindb)
            cudescc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            m,k  = transa == 'N' ? A.dims : (A.dims[2],A.dims[1])
            kB,n = transb == 'N' ? B.dims : (B.dims[2],B.dims[1])
            if( (k != kB) )
                throw(DimensionMismatch(""))
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
            C = CudaSparseMatrixCSR($elty, rowPtrC, CudaArray(zeros(Cint,nnz)), CudaArray(zeros($elty,nnz)), nnz, (m,n))
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

## preconditioners

# csric0 - incomplete Cholesky factorization with no pivoting

for (fname,elty) in ((:cusparseScsric0, :Float32),
                     (:cusparseDcsric0, :Float64),
                     (:cusparseCcsric0, :Complex64),
                     (:cusparseZcsric0, :Complex128))
    @eval begin
        function csric0!(transa::SparseChar,
                         typea::SparseChar,
                         A::CudaSparseMatrixCSR{$elty},
                         info::cusparseSolveAnalysisInfo_t,
                         index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cutype = cusparsetype(typea)
            cudesc = cusparseMatDescr_t(cutype, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, cusparseSolveAnalysisInfo_t),
                              cusparsehandle[1], cutransa, m, &cudesc, A.nzVal,
                              A.rowPtr, A.colVal, info))
            A
        end
        function csric0(transa::SparseChar,
                        typea::SparseChar,
                        A::CudaSparseMatrixCSR{$elty},
                        info::cusparseSolveAnalysisInfo_t,
                        index::SparseChar)
            csric0!(transa,typea,copy(A),info,index)
        end
    end
end

# csrilu0 - incomplete LU factorization with no pivoting

for (fname,elty) in ((:cusparseScsrilu0, :Float32),
                     (:cusparseDcsrilu0, :Float64),
                     (:cusparseCcsrilu0, :Complex64),
                     (:cusparseZcsrilu0, :Complex128))
    @eval begin
        function csrilu0!(transa::SparseChar,
                          A::CudaSparseMatrixCSR{$elty},
                          info::cusparseSolveAnalysisInfo_t,
                          index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, cusparseSolveAnalysisInfo_t),
                              cusparsehandle[1], cutransa, m, &cudesc, A.nzVal,
                              A.rowPtr, A.colVal, info))
            A
        end
        function csrilu0(transa::SparseChar,
                         A::CudaSparseMatrixCSR{$elty},
                         info::cusparseSolveAnalysisInfo_t,
                         index::SparseChar)
            csrilu0!(transa,copy(A),info,index)
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
