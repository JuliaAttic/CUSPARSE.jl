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

# type conversion
for (fname,elty) in ((:cusparseScsr2csc, :Float32),
                     (:cusparseDcsr2csc, :Float64),
                     (:cusparseCcsr2csc, :Complex64),
                     (:cusparseZcsr2csc, :Complex128))
    @eval begin
        function convert{$elty}(::Type{CudaSparseMatrixCSC{$elty}},
                                csr::CudaSparseMatrixCSR{$elty})
            cuind = cusparseindex('O')
            m,n = csr.dims
            csc = CudaSparseMatrixCSC($elty, zeros(Cint,n+1),zeros(Cint,csr.nnz),zeros($elty,csr.nnz),(m,n))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, Cint, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, cusparseAction_t, cusparseIndexBase_t),
                               cusparsehandle[1], m, n, csr.nnz, csr.nzVal,
                               csr.rowPtr, csr.colVal, csc.nzVal, csc.rowVal,
                               csc.colPtr, CUSPARSE_ACTION_NUMERIC, cuind))
            csc
        end
        function convert{$elty}(::Type{CudaSparseMatrixCSR{$elty}},
                                csc::CudaSparseMatrixCSC{$elty})
            cuind = cusparseindex('O')
            m,n = csc.dims
            println(csc.nnz)
            csr = CudaSparseMatrixCSR($elty, zeros(Cint,m+1),zeros(Cint,csc.nnz),zeros($elty,csc.nnz),(m,n))
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, Cint, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, cusparseAction_t, cusparseIndexBase_t),
                               cusparsehandle[1], n, m, csc.nnz, csc.nzVal,
                               csc.colPtr, csc.rowVal, csr.nzVal, csr.colVal,
                               csr.rowPtr, CUSPARSE_ACTION_NUMERIC, cuind))
            println(csr.nnz)
            csr
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

## level 3 functions

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
