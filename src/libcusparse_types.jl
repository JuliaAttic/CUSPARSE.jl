#enum cusparseStatus_t
#error messages from CUSPARSE

typealias cusparseStatus_t UInt32
const CUSPARSE_STATUS_SUCCESS                   = 0
const CUSPARSE_STATUS_NOT_INITIALIZED           = 1
const CUSPARSE_STATUS_ALLOC_FAILED              = 2
const CUSPARSE_STATUS_INVALID_VALUE             = 3
const CUSPARSE_STATUS_ARCH_MISMATCH             = 4
const CUSPARSE_STATUS_MAPPING_ERROR             = 5
const CUSPARSE_STATUS_EXECUTION_FAILED          = 6
const CUSPARSE_STATUS_INTERNAL_ERROR            = 7
const CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8

#enum cusparseAction_t
# perform operation only indices only or
# on both data and indices 
typealias cusparseAction_t UInt32
const CUSPARSE_ACTION_SYMBOLIC = 0
const CUSPARSE_ACTION_NUMERIC  = 1

#enum cusparseDirection_t
# parse dense matrix by rows or cols
# to compute its number of non-zeros
typealias cusparseDirection_t UInt32
const CUSPARSE_DIRECTION_ROW = 0
const CUSPARSE_DIRECTION_COL = 1

#enum cusparseHybPartition_t
# how to partition the HYB matrix
typealias cusparseHybPartition_t UInt32
const CUSPARSE_HYB_PARTITION_AUTO = 0
const CUSPARSE_HYB_PARTITION_USER = 1
const CUSPARSE_HYB_PARTITION_MAX  = 2

#enum cusparseFillMode_t
# filling for HE,TR,SY matrices
typealias cusparseFillMode_t UInt32
const CUSPARSE_FILL_MODE_LOWER = 0
const CUSPARSE_FILL_MODE_UPPER = 1

#enum cusparseDiagType_t
# is the diagonal all ones
typealias cusparseDiagType_t UInt32
const CUSPARSE_DIAG_TYPE_NON_UNIT = 0
const CUSPARSE_DIAG_TYPE_UNIT     = 1

#enum cusparsePointerMode_t
# are scalars on the host or device
typealias cusparsePointerMode_t UInt32
const CUSPARSE_POINTER_MODE_HOST   = 0
const CUSPARSE_POINTER_MODE_DEVICE = 1

#enum cusparseOperation_t
# no op, transpose, conj transpose
typealias cusparseOperation_t UInt32
const CUSPARSE_OPERATION_NON_TRANSPOSE       = 0
const CUSPARSE_OPERATION_TRANSPOSE           = 1
const CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2

#enum cusparseMatrixType_t
# marker to flag HE,TR,SY matrices
typealias cusparseMatrixType_t UInt32
const CUSPARSE_MATRIX_TYPE_GENERAL    = 0
const CUSPARSE_MATRIX_TYPE_SYMMETRIC  = 1
const CUSPARSE_MATRIX_TYPE_HERMITIAN  = 2
const CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3

#enum cusparseSolvePolicy_t
# keep level info in solvers
typealias cusparseSolvePolicy_t UInt32
const CUSPARSE_SOLVE_POLICY_NO_LEVEL  = 0
const CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1

#enum cusparseIndexBase_t
# 0- or 1-based indexing
typealias cusparseIndexBase_t UInt32
const CUSPARSE_INDEX_BASE_ZERO = 0
const CUSPARSE_INDEX_BASE_ONE  = 1

#struct cusparseMatDescr_t
# Describes shape and properties
# of a CUSPARSE matrix
type cusparseMatDescr_t
    MatrixType::cusparseMatrixType_t
    FillMode::cusparseFillMode_t
    DiagType::cusparseDiagType_t
    IndexBase::cusparseIndexBase_t
    function cusparseMatDescr_t(MatrixType,FillMode,DiagType,IndexBase)
        new(MatrixType,FillMode,DiagType,IndexBase)
    end
end

typealias cusparseSolveAnalysisInfo_t Ptr{Void}
typealias bsrsm2Info_t Ptr{Void}
typealias bsrsv2Info_t Ptr{Void}
typealias csrsv2Info_t Ptr{Void}
typealias csric02Info_t Ptr{Void}
typealias csrilu02Info_t Ptr{Void}
typealias bsric02Info_t Ptr{Void}
typealias bsrilu02Info_t Ptr{Void}

typealias cusparseContext Void
typealias cusparseHandle_t Ptr{cusparseContext}

#complex numbers

typealias cuComplex Complex{Float32}
typealias cuDoubleComplex Complex{Float64}

typealias CusparseFloat Union{Float64,Float32,Complex128,Complex64}
typealias CusparseReal Union{Float64,Float32}
typealias CusparseComplex Union{Complex128,Complex64}
