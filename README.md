# CUSPARSE.jl

[![Build Status](https://travis-ci.org/kshyatt/CUSPARSE.jl.svg?branch=master)](https://travis-ci.org/kshyatt/CUSPARSE.jl)
[![Coverage Status](https://coveralls.io/repos/kshyatt/CUSPARSE.jl/badge.svg?branch=master)](https://coveralls.io/r/kshyatt/CUSPARSE.jl?branch=master)
[![codecov.io](http://codecov.io/github/kshyatt/CUSPARSE.jl/coverage.svg?branch=master)](http://codecov.io/github/kshyatt/CUSPARSE.jl?branch=master)

Julia bindings for the [NVIDIA CUSPARSE](http://docs.nvidia.com/cuda/cusparse/) library. CUSPARSE is a high-performance sparse matrix linear algebra library.

# Table of Contents

- [Introduction](https://github.com/kshyatt/CUSPARSE.jl#introduction)
- [Current Features](https://github.com/kshyatt/CUSPARSE.jl#current-features)
- [Working with CUSPARSE.jl](https://github.com/kshyatt/CUSPARSE.jl#working-with-cusparsejl)
- [Example](https://github.com/kshyatt/CUSPARSE.jl#example)
- [When is CUPSARSE useful?](https://github.com/kshyatt/CUSPARSE.jl#when-is-cusparse-useful)
- [Contributing](https://github.com/kshyatt/CUSPARSE.jl#contributing)

# Introduction

CUSPARSE.jl proves bindings to a subset of the CUSPARSE library. It extends the amazing [CUDArt.jl](https://github.com/JuliaGPU/CUDArt.jl) library to provide four new sparse matrix classes:
    
- `CudaSparseMatrixCSC`

- `CudaSparseMatrixCSR`

- `CudaSparseMatrixBSR`

- `CudaSparseMatrixHYB`

which implement compressed sparse row/column storage, block CSR, and NVIDIA hybrid (`HYB`) `COO`-`ELL` format on the GPU. Since the native sparse type in Julia is `CSC`, and in CUSPARSE is `CSR`, automatic format conversion is provided, so that when you write
```julia
A = sprand(10,8,0.2)
d_A = CudaSparseMatrixCSR(A)
```
`A` is transformed into `CSC` format moved to the GPU, then auto-converted to `CSR` format for you. Thus, `d_A` is *not* a transpose of `A`! Similarly, if you have a matrix in dense format on the GPU (in a `CudaArray`), you can simply call `sparse` to turn it into a sparse representation. Right now `sparse` by default turns the matrix it is given into `CSR` format. It takes an optional argument that lets you select `CSC` or `HYB`:

```julia
d_A = CudaArray(rand(10,20))
d_A = sparse(d_A) #now in CSR format

d_B = CudaArray(rand(10,20))
d_B = sparse(d_B,'C') #now in CSC format

d_C = CudaArray(rand(10,20))
d_C = sparse(d_C,'H') #now in HYB format

d_D = CudaArray(rand(10,20))
d_D = sparse(d_C,'B') #now in BSR format
```
# Current Features

CUSPARSE.jl currently supports a subset of all the CUSPARSE functionality. What is implemented right now:
- [ ] Formats
    - [x] `CSR`
    - [x] `CSC`
    - [ ] `COO`
    - [ ] `ELL`
    - [x] `HYB`
    - [x] `BSR`
    - [ ] `BSRX`
- [x] Level 1 functions
    - [x] `axpyi`
    - [x] `doti`
    - [x] `dotci`
    - [x] `gthr`
    - [x] `gthrz`
    - [x] `roti`
    - [x] `sctr`
- [ ] Level 2 functions
    - [x] `bsrmv`
    - [ ] `bsrxmv`
    - [x] `csrmv`
    - [x] `bsrsv2_bufferSize`
    - [x] `bsrsv2_analysis`
    - [x] `bsrsv2_solve`
    - [x] `bsrsv2_zeroPivot`
    - [x] `csrsv_analysis`
    - [x] `csrsv_solve`
    - [x] `csrsv2_bufferSize`
    - [x] `csrsv2_analysis`
    - [x] `csrsv2_solve`
    - [x] `csrsv2_zeroPivot`
    - [x] `hybmv`
    - [x] `hybsv_analysis`
    - [x] `hybsv_solve`
- [x] Level 3 functions
    - [x] `csrmm`
    - [x] `csrmm2`
    - [x] `csrsm_analysis`
    - [x] `csrsm_solve`
    - [x] `bsrmm`
    - [x] `bsrsm2_bufferSize`
    - [x] `bsrsm2_analysis`
    - [x] `bsrsm2_solve`
    - [x] `bsrsm2_zeroPivot`
- [ ] Extensions
    - [x] `csrgeam`
    - [x] `csrgemm`
    - [ ] `csrgemm2`
- [ ] Preconditioners
    - [x] `csric0`
    - [x] `csric02_bufferSize`
    - [x] `csric02_analysis`
    - [x] `csric02`
    - [x] `csric02_zeroPivot`
    - [x] `csrilu0`
    - [ ] `csrilu02_numericBoost`
    - [x] `csrilu02_bufferSize`
    - [x] `csrilu02_analysis`
    - [x] `csrilu02`
    - [x] `csrilu02_zeroPivot`
    - [x] `bsric02_bufferSize`
    - [x] `bsric02_analysis`
    - [x] `bsric02`
    - [x] `bsric02_zeroPivot`
    - [ ] `bsrilu02_numericBoost`
    - [x] `bsrilu02_bufferSize`
    - [x] `bsrilu02_analysis`
    - [x] `bsrilu02`
    - [x] `bsrilu02_zeroPivot`
    - [x] `gtsv`
    - [x] `gtsv_noPivot`
    - [x] `gtsvStridedBatch`
- [ ] Type conversions
    - [x] `bsr2csr`
    - [ ] `gebsr2gebsc_bufferSize`
    - [ ] `gebsr2gebsc`
    - [ ] `gebsc2gebsr_bufferSize`
    - [ ] `gebsc2gebsr`
    - [ ] `gebsr2csr`
    - [ ] `csr2gebsr_bufferSize`
    - [ ] `csr2gebsr`
    - [ ] `coo2csr`
    - [x] `csc2dense`
    - [x] `csc2hyb`
    - [x] `csr2bsr`
    - [ ] `csr2coo`
    - [x] `csr2csc`
    - [x] `csr2dense`
    - [x] `csr2hyb`
    - [x] `dense2csc`
    - [x] `dense2csr`
    - [x] `dense2hyb`
    - [x] `hyb2csc`
    - [x] `hyb2csr`
    - [x] `hyb2dense`
    - [x] `nnz`
    - [ ] `CreateIdentityPermutation`
    - [ ] `coosort`
    - [ ] `csrsort`
    - [ ] `cscsort`
    - [ ] `csru2csr`

This is a big, ugly looking list. The actual operations CUSPARSE.jl supports are:

- Dense Vector + a * Sparse Vector
- Sparse Vector dot Dense Vector
- Scatter Sparse Vector into Dense Vector
- Gather Dense Vector into Sparse Vector
- Givens Rotation on Sparse and Dense Vectors
- Sparse Matrix * Dense Vector
- Sparse Matrix \ Dense Vector
- Sparse Matrix \ Dense Vector
- Sparse Matrix * Dense Matrix
- Sparse Matrix * Sparse Matrix
- Sparse Matrix + Sparse Matrix
- Sparse Matrix \ Dense Matrix
- Incomplete LU factorization with 0 pivoting
- Incomplete Cholesky factorization with 0 pivoting
- Tridiagonal Matrix \ Dense Vector

## A note about factorizations

CUSPARSE provides **incomplete** LU and Cholesky factorization. Often, for a sparse matrix, the full LU or Cholesky factorization is much less sparse than the original matrix.
This is a problem if the sparse matrix is very large, since GPU memory is limited. CUSPARSE provides **incomplete** versions of these factorizations, such that `A` is
**approximatily** equal to `L * U` or `L* L`. In particular, the incomplete factorizations have the same sparsity pattern as `A`, so that they occupy the same amount of GPU
memory. They are preconditioners - we can solve the problem `y = A \ x` by applying them iteratively. You should not expect `ilu0` or `ic0` in CUSPARSE to have matrix elements
equal to those from Julia `lufact` or `cholfact`, because the Julia factorizations are complete.

## Type Conversions

| From\To: | Dense   | CSR              | CSC              | BSR           | HYB              |
|----------|---------|------------------|------------------|---------------|------------------|
| **Dense**| N/A     |`sparse(A)`       |`sparse(A,'C')`   |`sparse(A,'B')`|`sparse(A,'H')`   |
| **CSR**  |`full(A)`| N/A              |`switch2csr(A)`   |`switch2csr(A)`|`switch2csr(A)`   |
| **CSC**  |`full(A)`|`switch2csc(A)`   | N/A              |`switch2csc(A)`|`switch2csc(A)`   |
| **BSR**  |`full(A)`|`switch2bsr(A,bD)`|`switch2bsr(A,bD)`| N/A           |`switch2bsr(A,bD)`|
| **HYB**  |`full(A)`|`switch2hyb(A)`   |`switch2hyb(A)`   |`switch2hyb(A)`| N/A              |

# Working with CUSPARSE.jl

CUSPARSE.jl exports its matrix types, so you do not have to prepend them with anything. To use a CUSPARSE function, just
```julia
using CUSPARSE

### stuff happens here

CUSPARSE.csrmv( #arguments! )
```
**Important Note:** CUSPARSE solvers (`sv`, `sm`) assume the matrix you are solving is **triangular**. If you pass them a general matrix you will get the **wrong** answer!

# Example
A simple example of creating two sparse matrices `A`,`B` on the CPU, moving them to the GPU, adding them, and bringing the result back:

```julia
using CUSPARSE

# dimensions and fill proportion
N = 20
M = 10
p = 0.1

# create matrices A,B on the CPU 
A = sprand(N,M,p)
B = sprand(N,M,p)

# convert A,B to CSR format and
# move them to the GPU - one step
d_A = CudaSparseMatrixCSR(A)
d_B = CudaSparseMatrixCSR(B)

# generate scalar parameters
alpha = rand()
beta  = rand()

# perform alpha * A + beta * B
d_C = CUSPARSE.geam(alpha, d_A, beta, d_B, 'O', 'O', 'O')

# bring the result back to the CPU
C = CUSPARSE.to_host(d_C)

# observe a zero matrix
alpha*A + beta*B - C
```

Some questions you might have:
- What are the three `'O'`s for?
    - CUSPARSE allows us to use one- or zero-based indexing. Julia uses one-based indexing for arrays, but many other libraries (for instance, C-based libraries) use zero-based. The `'O'`s tell CUSPARSE that our matrices are one-based. If you had a zero-based matrix from an external library, you can tell CUSPARSE using `'Z'`.
- Should we move `alpha` and `beta` to the GPU?
    - We do not have to. CUSPARSE can read in scalar parameters like `alpha` and `beta` from the host (CPU) memory. You can just pass them to the function and CUSPARSE.jl handles telling the CUDA functions where they are for you. If you have an array, like `A` and `B`, you do need to move it to the GPU before CUSPARSE can work on it. Similarly, to see results, if they are in array form, you will need to move them back to the CPU with `to_host`.
- Since `d_C` is in `CSR` format, is `C` the transpose of what we want?
    - No. CUSPARSE.jl handles the conversion internally so that the final result is in `CSC` format for Julia, and *not* the transpose of the correct answer.

# When is CUSPARSE useful?

Moving data between the CPU and GPU memory is very time-intensive. In general, if you only do one operation on the GPU (e.g. one matrix-vector multiplication), the computation is dominated by the time spent copying data. However, if you do many operations with the data you have on the GPU, like doing twenty matrix-vector multiplications, then the GPU can easily beat the CPU. Below you can see some timing tests for the CPU vs the GPU for 20 operations:
![matrix matrix multiplication](/test/mm.png)
![matrix vector multiplication](/test/mv.png)
![matrix vector solve](/test/sv.png)

The GPU does very well in these tests, but if we only did one operation, the GPU would do as well as or worse than the CPU. It is not worth it to use the GPU if most of your time will be spent copying data around!

# Contributing

Contributions are very welcome! If you write wrappers for one of the CUSPARSE functions, please include some tests in `test/runtests.jl` for your wrapper. Ideally test each of the types the function you wrap can accept, e.g. `Float32`, `Float64`, and possibly `Complex64`, `Complex128`.
