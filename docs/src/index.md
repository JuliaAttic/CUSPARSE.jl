# CUSPARSE.jl

### Container types
```@docs
CudaSparseVector
CudaSparseMatrixCSC
CudaSparseMatrixCSR
CudaSparseMatrixBSR
CudaSparseMatrixHYB
CudaSparseMatrix
CUSPARSE.CompressedSparse
```

### Utility types
```@docs
CUSPARSE.cusparseMatDescr_t
CUSPARSE.cusparseDirection_t
CUSPARSE.cusparseFillMode_t
CUSPARSE.cusparseSolveAnalysisInfo_t
CUSPARSE.cusparseSolvePolicy_t
CUSPARSE.cusparseStatus_t
CUSPARSE.cusparseDiagType_t
CUSPARSE.cusparseMatrixType_t
CUSPARSE.cusparseIndexBase_t
CUSPARSE.cusparseHybPartition_t
CUSPARSE.cusparseOperation_t
CUSPARSE.cusparsePointerMode_t
CUSPARSE.cusparseAction_t
```

### Utility functions
```@docs
CUSPARSE.cusparsediag
CUSPARSE.cusparsedir
CUSPARSE.cusparsefill
CUSPARSE.cusparseindex
CUSPARSE.chkmmdims
CUSPARSE.statuscheck
CUSPARSE.cusparseop
CUSPARSE.chkmvdims
CUSPARSE.getDescr
CUSPARSE.cusparsetype
```

### Type Conversions
```@docs
switch2csc
switch2csr
switch2bsr
switch2hyb
```

### Level 1 Functions
```@docs
axpyi!
doti!
dotci!
gthr!
gthrz!
roti!
sctr!
```

### Level 2 Functions
```@docs
mv!
sv2!
sv_analysis
sv_solve!
sv
```

### Level 3 Functions
```@docs
mm2!
mm!
sm_analysis
sm_solve
sm
```

### Extensions
```@docs
geam
gemm
```

### Preconditioners
```@docs
ic0!
ic02!
ilu0!
ilu02!
gtsv!
gtsv_nopivot!
gtsvStridedBatch!
```
