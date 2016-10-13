# CUSPARSE.jl

### Container types

Analogous to `CUDArt.jl` and its `CudaArray` types, you can use
a variety of sparse matrix containers similar to those in Julia
base.

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

These types are used to give CUSPARSE internals information
about the matrices they are working on. Most are opaque pointers
or `enums`.

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

These functions allow you to switch between different sparse
matrix formats. This is particularly useful because Julia
uses a compressed sparse **column** representation internally,
while most CUSPARSE routines work with compressed sparse **row**
matrices (or variants of them, such as the `BSR` and `HYB` formats).

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

These are in-place incomplete preconditioners for a linear solve.
They are in-place in the sense that doing a full Cholesky/LU solve
would require more memory than the starting matrix uses, and this
may not be possible on the restricted memory of the GPU. Instead,
an incomplete solution is performed, where the structure of the
matrix is untouched (so it occupies the same amount of memory)
but its values are updated. 

```@docs
ic0!
ic02!
ilu0!
ilu02!
gtsv!
gtsv_nopivot!
gtsvStridedBatch!
```
