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

# Level 1 CUSPARSE functions

for (fname,elty) in ((:cusparseSaxpyi, :Float32),
                     (:cusparseDaxpyi, :Float64),
                     (:cusparseCaxpyi, :Complex64),
                     (:cusparseZaxpyi, :Complex128))
    @eval begin
        function axpyi!(alpha::$elty,
                        X::CudaSparseMatrix{$elty},
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
                       X::CudaSparseMatrix{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            axpyi!(alpha,X,copy(Y),index)
        end
        function axpyi(X::CudaSparseMatrix{$elty},
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
        function $jname(X::CudaSparseMatrix{$elty},
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
        function gthr!(X::CudaSparseMatrix{$elty},
                      Y::CudaVector{$elty},
                      index::SparseChar)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{$elty},
                               Ptr{Cint}, cusparseIndexBase_t), cusparsehandle[1],
                              X.nnz, Y, X.nzVal, X.rowVal, cuind))
            X
        end
        function gthr(X::CudaSparseMatrix{$elty},
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
        function gthrz!(X::CudaSparseMatrix{$elty},
                        Y::CudaVector{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, Ptr{$elty},
                               Ptr{Cint}, cusparseIndexBase_t), cusparsehandle[1],
                              X.nnz, Y, X.nzVal, X.rowVal, cuind))
            X,Y
        end
        function gthrz(X::CudaSparseMatrix{$elty},
                       Y::CudaVector{$elty},
                       index::SparseChar)
            gthrz!(copy(X),copy(Y),index)
        end
    end
end

for (fname,elty) in ((:cusparseSroti, :Float32),
                     (:cusparseDroti, :Float64))
    @eval begin
        function roti!(X::CudaSparseMatrix{$elty},
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
        function roti(X::CudaSparseMatrix{$elty},
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
        function sctr!(X::CudaSparseMatrix{$elty},
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
        function sctr(X::CudaSparseMatrix{$elty},
                      index::SparseChar)
            sctr!(X,CudaArray(zeros($elty,X.dims[1])),index)
        end
    end
end
