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
function cusparseCreateHybMat(hybA)
  statuscheck(ccall( (:cusparseCreateHybMat, libcusparse), cusparseStatus_t, (Ptr{cusparseHybMat_t},), hybA))
end
function cusparseDestroyHybMat(hybA)
  statuscheck(ccall( (:cusparseDestroyHybMat, libcusparse), cusparseStatus_t, (cusparseHybMat_t,), hybA))
end
function cusparseCreateSolveAnalysisInfo(info)
  statuscheck(ccall( (:cusparseCreateSolveAnalysisInfo, libcusparse), cusparseStatus_t, (Ptr{cusparseSolveAnalysisInfo_t},), info))
end
function cusparseDestroySolveAnalysisInfo(info)
  statuscheck(ccall( (:cusparseDestroySolveAnalysisInfo, libcusparse), cusparseStatus_t, (cusparseSolveAnalysisInfo_t,), info))
end
function cusparseCreateBsrsm2Info(info)
  statuscheck(ccall( (:cusparseCreateBsrsm2Info, libcusparse), cusparseStatus_t, (Ptr{bsrsm2Info_t},), info))
end
function cusparseDestroyBsrsm2Info(info)
  statuscheck(ccall( (:cusparseDestroyBsrsm2Info, libcusparse), cusparseStatus_t, (bsrsm2Info_t,), info))
end
function cusparseCreateBsrsv2Info(info)
  statuscheck(ccall( (:cusparseCreateBsrsv2Info, libcusparse), cusparseStatus_t, (Ptr{bsrsv2Info_t},), info))
end
function cusparseDestroyBsrsv2Info(info)
  statuscheck(ccall( (:cusparseDestroyBsrsv2Info, libcusparse), cusparseStatus_t, (bsrsv2Info_t,), info))
end
function cusparseCreateCsrsv2Info(info)
  statuscheck(ccall( (:cusparseCreateCsrsv2Info, libcusparse), cusparseStatus_t, (Ptr{csrsv2Info_t},), info))
end
function cusparseDestroyCsrsv2Info(info)
  statuscheck(ccall( (:cusparseDestroyCsrsv2Info, libcusparse), cusparseStatus_t, (csrsv2Info_t,), info))
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

function cusparseSbsrmv(handle, dir, transA, mb, nb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y)
  statuscheck(ccall( (:cusparseSbsrmv, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_T, cusparseOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}), handle, dir, mb, nb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y))
end
function cusparseDbsrmv(handle, dir, transA, mb, nb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y)
  statuscheck(ccall( (:cusparseDbsrmv, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_T, cusparseOperation_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), handle, dir, mb, nb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y))
end
function cusparseCbsrmv(handle, dir, transA, mb, nb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y)
  statuscheck(ccall( (:cusparseCbsrmv, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_T, cusparseOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Ptr{cuComplex}), handle, dir, mb, nb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y))
end
function cusparseZbsrmv(handle, dir, transA, mb, nb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y)
  statuscheck(ccall( (:cusparseZbsrmv, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_T, cusparseOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}), handle, dir, mb, nb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y))
end
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

function cusparseSbsrmm(handle, dir, transA, transB, mb, n, kb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseSbsrmm, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, dir, transA, transB, mb, n, kb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc))
end
function cusparseDbsrmm(handle, dir, transA, transB, mb, n, kb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseDbsrmm, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, dir, transA, transB, mb, n, kb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc))
end
function cusparseCbsrmm(handle, dir, transA, transB, mb, n, kb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseCbsrmm, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, dir, transA, transB, mb, n, kb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc))
end
function cusparseZbsrmm(handle, dir, transA, transB, mb, n, kb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseZbsrmm, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, dir, transA, transB, mb, n, kb, nnz, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc))
end
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
  statuscheck(ccall( (:cusparseScsrmm2, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc))
end
function cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseDcsrmm2, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc))
end
function cusparseCcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseCcsrmm2, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc))
end
function cusparseZcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
  statuscheck(ccall( (:cusparseDcsrmm2, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc))
end

# extensions

function cusparseXcsrgeamNnz(handle, m, n, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, csrRowPtrC, nnzTotalDevHostPtr)
  statuscheck(ccall( (:cusparseXcsrgeamNnz, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{Cint}, Ptr{Cint}), handle, m, n, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, csrRowPtrC, nnzTotalDevHostPtr))
end
function cusparseScsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC)
  statuscheck(ccall( (:cusparseScsrgeam, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{Cfloat}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}), handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC))
end
function cusparseDcsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC)
  statuscheck(ccall( (:cusparseDcsrgeam, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{Cdouble}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}), handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC))
end
function cusparseCcsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC)
  statuscheck(ccall( (:cusparseCcsrgeam, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cuComplex}, Ptr{cusparseMatDescr_t}, Cint, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Ptr{cusparseMatDescr_t}, Cint, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}), handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC))
end
function cusparseZcsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC)
  statuscheck(ccall( (:cusparseZcsrgeam, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cusparseMatDescr_t}, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{cusparseMatDescr_t}, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}), handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC))
end
function cusparseXcsrgemmNnz(handle, transa, transb, m, n, k, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, csrRowPtrC, nnzTotalDevHostPtr)
  statuscheck(ccall( (:cusparseXcsrgemmNnz, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{Cint}, Ptr{Cint}), handle, transa, transb, m, n, k, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, csrRowPtrC, nnzTotalDevHostPtr))
end
function cusparseScsrgemm(handle, transa, transb, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC)
  statuscheck(ccall( (:cusparseScsrgemm, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}), handle, transa, transb, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC))
end
function cusparseDcsrgemm(handle, transa, transb, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC)
  statuscheck(ccall( (:cusparseDcsrgemm, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}), handle, transa, transb, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC))
end
function cusparseCcsrgemm(handle, transa, transb, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC)
  statuscheck(ccall( (:cusparseCcsrgemm, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{cusparseMatDescr_t}, Cint, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Cint, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}), handle, transa, transb, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC))
end
function cusparseZcsrgemm(handle, transa, transb, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC)
  statuscheck(ccall( (:cusparseZcsrgemm, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{cusparseMatDescr_t}, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Cint, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}), handle, transa, transb, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC))
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
function cusparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda)
  statuscheck(ccall( (:cusparseScsr2dense, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Ptr{Cint}), handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda))
end
function cusparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda)
  statuscheck(ccall( (:cusparseDcsr2dense, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}), handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda))
end
function cusparseCcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda)
  statuscheck(ccall( (:cusparseCcsr2dense, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Ptr{Cint}), handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda))
end
function cusparseZcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda)
  statuscheck(ccall( (:cusparseZcsr2dense, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{Cint}), handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda))
end
function cusparseScsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda)
  statuscheck(ccall( (:cusparseScsc2dense, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Ptr{Cint}), handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda))
end
function cusparseDcsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda)
  statuscheck(ccall( (:cusparseDcsc2dense, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}), handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda))
end
function cusparseCcsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda)
  statuscheck(ccall( (:cusparseCcsc2dense, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Ptr{Cint}), handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda))
end
function cusparseZcsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda)
  statuscheck(ccall( (:cusparseZcsc2dense, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{Cint}), handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda))
end
function cusparseSdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA)
  statuscheck(ccall( (:cusparseSdense2csr, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}), handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA))
end
function cusparseDdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA)
  statuscheck(ccall( (:cusparseDdense2csr, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}), handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA))
end
function cusparseCdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA)
  statuscheck(ccall( (:cusparseCdense2csr, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}), handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA))
end
function cusparseZdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA)
  statuscheck(ccall( (:cusparseZdense2csr, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}), handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA))
end
function cusparseSdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA)
  statuscheck(ccall( (:cusparseSdense2csc, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}), handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA))
end
function cusparseDdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA)
  statuscheck(ccall( (:cusparseDdense2csc, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}), handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA))
end
function cusparseCdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA)
  statuscheck(ccall( (:cusparseCdense2csc, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}), handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA))
end
function cusparseZdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA)
  statuscheck(ccall( (:cusparseZdense2csc, libcusparse), cusparseStatus_t, (cusparseHandle_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}), handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA))
end
function cusparseSnnz(handle, dir, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr)
  statuscheck(ccall( (:cusparseSnnz, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr))
end
function cusparseDnnz(handle, dir, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr)
  statuscheck(ccall( (:cusparseDnnz, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr))
end
function cusparseCnnz(handle, dir, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr)
  statuscheck(ccall( (:cusparseCnnz, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr))
end
function cusparseZnnz(handle, dir, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr)
  statuscheck(ccall( (:cusparseZnnz, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr))
end
function cusparseXcsr2bsrNnz(handle, dir, m, n, descrA, rowPtrA, colIndA, blockDim, descrC, rowPtrC, nnzTotalDevHostPtr)
  statuscheck(ccall( (:cusparseXcsr2bsrNnz, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, rowPtrA, colIndA, blockDim, descrC, rowPtrC, nnzTotalDevHostPtr))
end
function cusparseScsr2bsr(handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC)
  statuscheck(ccall( (:cusparseScsr2bsr, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC))
end
function cusparseDcsr2bsr(handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC)
  statuscheck(ccall( (:cusparseDcsr2bsr, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC))
end
function cusparseCcsr2bsr(handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC)
  statuscheck(ccall( (:cusparseCcsr2bsr, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC))
end
function cusparseZcsr2bsr(handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC)
  statuscheck(ccall( (:cusparseZcsr2bsr, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC))
end
function cusparseSbsr2csr(handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC)
  statuscheck(ccall( (:cusparseSbsr2csr, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC))
end
function cusparseDbsr2csr(handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC)
  statuscheck(ccall( (:cusparseDbsr2csr, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{cusparseMatDescr_t}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC))
end
function cusparseCbsr2csr(handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC)
  statuscheck(ccall( (:cusparseCbsr2csr, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC))
end
function cusparseZbsr2csr(handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC)
  statuscheck(ccall( (:cusparseZbsr2csr, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{cusparseMatDescr_t}, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}), handle, dir, m, n, descrA, nzValA, rowPtrA, colIndA, blockDim, descrC, nzValC, rowPtrC, colIndC))
end
