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
