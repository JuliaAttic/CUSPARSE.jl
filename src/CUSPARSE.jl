module CUSPARSE

if VERSION >= v"0.5.0-dev+742"
    importall Base.SparseArrays
else
    importall Base.SparseMatrix
end

using CUDArt

typealias SparseChar Char
import Base.one
import Base.zero

export CudaSparseMatrixCSC, CudaSparseMatrixCSR,
       CudaSparseMatrixHYB, CudaSparseMatrixBSR,
       CudaSparseMatrix, AbstractCudaSparseMatrix,
       CudaSparseVector

include("util.jl")
include("libcusparse_types.jl")

function statusmessage( status )
    if status == CUSPARSE_STATUS_SUCCESS
        return "cusparse success"
    end
    if status == CUSPARSE_STATUS_NOT_INITIALIZED
        return "cusparse not initialized"
    end
    if status == CUSPARSE_STATUS_ALLOC_FAILED
        return "cusparse allocation failed"
    end
    if status == CUSPARSE_STATUS_INVALID_VALUE
        return "cusparse invalid value"
    end
    if status == CUSPARSE_STATUS_ARCH_MISMATCH
        return "cusparse architecture mismatch"
    end
    if status == CUSPARSE_STATUS_MAPPING_ERROR
        return "cusparse mapping error"
    end
    if status == CUSPARSE_STATUS_EXECUTION_FAILED
        return "cusparse execution failed"
    end
    if status == CUSPARSE_STATUS_INTERNAL_ERROR
        return "cusparse internal error"
    end
    if status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
        return "cusparse matrix type not supported"
    end
end

function statuscheck( status )
    if status == CUSPARSE_STATUS_SUCCESS
        return nothing
    end
    warn("CUSPARSE error triggered from:")
    Base.show_backtrace(STDOUT, backtrace())
    println()
    throw(statusmessage( status ))
end

cuda_versions = ["80", "75", "70", "65"]
lib_list = vcat(["libcusparse", "cusparse"], "cusparse64_" .* cuda_versions, "cusparse32_" .* cuda_versions)
const libcusparse = Libdl.find_library(lib_list, ["/usr/lib/", "/usr/local/cuda", ""])

if isempty(libcusparse)
    error("CUSPARSE library not found!")
end

include("libcusparse.jl")

#setup handler for cusparse

cusparsehandle = cusparseHandle_t[0]
cusparseCreate( cusparsehandle )

#clean up handle at exit
atexit( ()->cusparseDestroy(cusparsehandle[1]) )

include("sparse.jl")

end
