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
function cusparseCreateCsric02Info(info)
  statuscheck(ccall( (:cusparseCreateCsric02Info, libcusparse), cusparseStatus_t, (Ptr{csric02Info_t},), info))
end
function cusparseDestroyCsric02Info(info)
  statuscheck(ccall( (:cusparseDestroyCsric02Info, libcusparse), cusparseStatus_t, (csric02Info_t,), info))
end
function cusparseCreateCsrilu02Info(info)
  statuscheck(ccall( (:cusparseCreateCsrilu02Info, libcusparse), cusparseStatus_t, (Ptr{csrilu02Info_t},), info))
end
function cusparseDestroyCsrilu02Info(info)
  statuscheck(ccall( (:cusparseDestroyCsrilu02Info, libcusparse), cusparseStatus_t, (csrilu02Info_t,), info))
end
function cusparseCreateBsric02Info(info)
  statuscheck(ccall( (:cusparseCreateBsric02Info, libcusparse), cusparseStatus_t, (Ptr{bsric02Info_t},), info))
end
function cusparseDestroyBsric02Info(info)
  statuscheck(ccall( (:cusparseDestroyBsric02Info, libcusparse), cusparseStatus_t, (bsric02Info_t,), info))
end
function cusparseCreateBsrilu02Info(info)
  statuscheck(ccall( (:cusparseCreateBsrilu02Info, libcusparse), cusparseStatus_t, (Ptr{bsrilu02Info_t},), info))
end
function cusparseDestroyBsrilu02Info(info)
  statuscheck(ccall( (:cusparseDestroyBsrilu02Info, libcusparse), cusparseStatus_t, (bsrilu02Info_t,), info))
end
