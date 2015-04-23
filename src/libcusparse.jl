# Julia wrapper for header: /usr/local/cuda/include/cusparse.h

#helper functions
function cusparseCreate(handle)
  statuscheck(ccall( (:cusparseCreate, libcusparse), cusparseStatus_t, (Ptr{cusparseHandle_t},), handle))
end
function cusparseDestroy(handle)
  statuscheck(ccall( (:cusparseDestroy, libcusparse), cusparseStatus_t, (cusparseHandle_t,), handle))
end
function cusparseGetVersion(handle, version)
  statuscheck(ccall( (:cusparseGetVersion, libcusparse), cusparseStatus_t, (cusparseHandle_t, Ptr{Cint}), handle, version))
end
function cusparseSetStream(handle, streamId)
  statuscheck(ccall( (:cusparseSetStream, libcusparse), cusparseStatus_t, (cusparseHandle_t, cudaStream_t), handle, streamId))
end
function cusparseGetStream(handle, streamId)
  statuscheck(ccall( (:cusparseGetStream, libcusparse), cusparseStatus_t, (cusparseHandle_t, Ptr{cudaStream_t}), handle, streamId))
end
function cusparseGetPointerMode(handle, mode)
  statuscheck(ccall( (:cusparseGetPointerMode, libcusparse), cusparseStatus_t, (cusparseHandle_t, Ptr{cusparsePointerMode_t}), handle, mode))
end
function cusparseSetPointerMode(handle, mode)
  statuscheck(ccall( (:cusparseSetPointerMode, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparsePointerMode_t), handle, mode))
end

# level 1 functions
function cusparseSaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase ) 
  statuscheck(ccall( (:cusparseSaxpyi, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cfloat}, cusparseIndexBase_t), handle, nnz, alpha, xVal, xInd, y, idxBase))
end
function cusparseDaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase ) 
  statuscheck(ccall( (:cusparseDaxpyi, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, cusparseIndexBase_t), handle, nnz, alpha, xVal, xInd, y, idxBase))
end
function cusparseCaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase ) 
  statuscheck(ccall( (:cusparseCaxpyi, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Ptr{Cint}, Ptr{cuComplex}, cusparseIndexBase_t), handle, nnz, alpha, xVal, xInd, y, idxBase))
end
function cusparseDaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase ) 
  statuscheck(ccall( (:cusparseZaxpyi, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{cuDoubleComplex}, cusparseIndexBase_t), handle, nnz, alpha, xVal, xInd, y, idxBase))
end
function cusparseSdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase ) 
  statuscheck(ccall( (:cusparseSdoti, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cfloat}, Ptr{Cfloat}, cusparseIndexBase_t), handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase))
end
function cusparseDdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase ) 
  statuscheck(ccall( (:cusparseDdoti, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, cusparseIndexBase_t), handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase))
end
function cusparseCdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase ) 
  statuscheck(ccall( (:cusparseCdoti, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{cuComplex}, Ptr{Cint}, Ptr{cuComplex}, Ptr{cuComplex}, cusparseIndexBase_t), handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase))
end
function cusparseZdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase ) 
  statuscheck(ccall( (:cusparseZdoti, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, cusparseIndexBase_t), handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase))
end
function cusparseCdotci(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase ) 
  statuscheck(ccall( (:cusparseCdotci, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{cuComplex}, Ptr{Cint}, Ptr{cuComplex}, Ptr{cuComplex}, cusparseIndexBase_t), handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase))
end
function cusparseZdotci(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase ) 
  statuscheck(ccall( (:cusparseZdotci, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, cusparseIndexBase_t), handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase))
end
function cusparseSgthr(handle, nnz, y, xVal, xInd, idxBase ) 
  statuscheck(ccall( (:cusparseSgthr, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cint}, cusparseIndexBase_t), handle, nnz, y, xVal, xInd, idxBase))
end
function cusparseDgthr(handle, nnz, y, xVal, xInd, idxBase ) 
  statuscheck(ccall( (:cusparseDgthr, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, cusparseIndexBase_t), handle, nnz, y, xVal, xInd, idxBase))
end
function cusparseCgthr(handle, nnz, y, xVal, xInd, idxBase ) 
  statuscheck(ccall( (:cusparseCgthr, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Ptr{Cint}, cusparseIndexBase_t), handle, nnz, y, xVal, xInd, idxBase))
end
function cusparseZgthr(handle, nnz, y, xVal, xInd, idxBase ) 
  statuscheck(ccall( (:cusparseZgthr, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Ptr{Cint}, cusparseIndexBase_t), handle, nnz, y, xVal, xInd, idxBase))
end
function cusparseSgthrz(handle, nnz, y, xVal, xInd, idxBase ) 
  statuscheck(ccall( (:cusparseSgthrz, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cint}, cusparseIndexBase_t), handle, nnz, y, xVal, xInd, idxBase))
end
function cusparseDgthrz(handle, nnz, y, xVal, xInd, idxBase ) 
  statuscheck(ccall( (:cusparseDgthrz, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, cusparseIndexBase_t), handle, nnz, y, xVal, xInd, idxBase))
end
function cusparseCgthrz(handle, nnz, y, xVal, xInd, idxBase ) 
  statuscheck(ccall( (:cusparseCgthrz, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Ptr{Cint}, cusparseIndexBase_t), handle, nnz, y, xVal, xInd, idxBase))
end
function cusparseZgthrz(handle, nnz, y, xVal, xInd, idxBase ) 
  statuscheck(ccall( (:cusparseZgthrz, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Ptr{Cint}, cusparseIndexBase_t), handle, nnz, y, xVal, xInd, idxBase))
end
function cusparseSroti(handle, nnz, xVal, xInd, y, c, s, idxBase ) 
  statuscheck(ccall( (:cusparseSroti, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, cusparseIndexBase_t), handle, nnz, xVal, xInd, y, c, s, idxBase))
end
function cusparseDroti(handle, nnz, xVal, xInd, y, c, s, idxBase ) 
  statuscheck(ccall( (:cusparseDroti, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, cusparseIndexBase_t), handle, nnz, xVal, xInd, y, c, s, idxBase))
end
function cusparseSsctr(handle, nnz, xVal, xInd, y, idxBase ) 
  statuscheck(ccall( (:cusparseSsctr, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cfloat}, cusparseIndexBase_t), handle, nnz, xVal, xInd, y, idxBase))
end
function cusparseDsctr(handle, nnz, xVal, xInd, y, idxBase ) 
  statuscheck(ccall( (:cusparseDsctr, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, cusparseIndexBase_t), handle, nnz, xVal, xInd, y, idxBase))
end
function cusparseCsctr(handle, nnz, xVal, xInd, y, idxBase ) 
  statuscheck(ccall( (:cusparseCsctr, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{cuComplex}, Ptr{Cint}, Ptr{cuComplex}, cusparseIndexBase_t), handle, nnz, xVal, xInd, y, idxBase))
end
function cusparseZsctr(handle, nnz, xVal, xInd, y, idxBase ) 
  statuscheck(ccall( (:cusparseZsctr, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{cuDoubleComplex}, cusparseIndexBase_t), handle, nnz, xVal, xInd, y, idxBase))
end

# level 2 functions

function cusparseScsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y)
  statuscheck(ccall( (:cusparseScsrmv, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}), handle, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y))
end
function cusparseDcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y)
  statuscheck(ccall( (:cusparseDcsrmv, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), handle, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y))
end
function cusparseCcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y)
  statuscheck(ccall( (:cusparseCcsrmv, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Ptr{cuComplex}, Ptr{cuComplex}), handle, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y))
end
function cusparseZcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y)
  statuscheck(ccall( (:cusparseZcsrmv, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}), handle, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y))
end

# level 3 functions

function cusparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseScsrmm, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc))
end
function cusparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseDcsrmm, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc))
end
function cusparseCcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseCcsrmm, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc))
end
function cusparseZcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseZcsrmm, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc))
end
function cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseScsrmm2, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc))
end
function cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseDcsrmm2, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc))
end
function cusparseCcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseCcsrmm2, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc))
end
function cusparseZcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseDcsrmm2, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc))
end

# type conversion
function cusparseScsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase)
  statuscheck(ccall( (:cusparseScsr2csc, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, cusparseAction_t, cusparseIndexBase_t), handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase))
end
function cusparseDcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase)
  statuscheck(ccall( (:cusparseDcsr2csc, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, cusparseAction_t, cusparseIndexBase_t), handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase))
end
function cusparseCcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase)
  statuscheck(ccall( (:cusparseCcsr2csc, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, cusparseAction_t, cusparseIndexBase_t), handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase))
end
function cusparseZcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase)
  statuscheck(ccall( (:cusparseZcsr2csc, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cusparseAction_t, cusparseIndexBase_t), handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase))
end
