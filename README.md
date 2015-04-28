# CUSPARSE.jl

[![Build Status](https://travis-ci.org/kshyatt/CUSPARSE.jl.svg?branch=master)](https://travis-ci.org/kshyatt/CUSPARSE.jl)

Julia bindings for the [NVIDIA CUSPARSE](http://docs.nvidia.com/cuda/cusparse/) library. CUSPARSE is a high-performance sparse matrix linear algebra library.

# Table of Contents

- [Introduction](https://github.com/kshyatt/CUSPARSE.jl#introduction)
- [Current Features](https://github.com/kshyatt/CUSPARSE.jl#current-features)
- [Working with CUSPARSE.jl](https://github.com/kshyatt/CUSPARSE.jl#working-with-cusparsejl)
- [Example](https://github.com/kshyatt/CUSPARSE.jl#example)
- [Contributing](https://github.com/kshyatt/CUSPARSE.jl#contributing)

# Introduction

CUSPARSE.jl proves bindings to a subset of the CUSPARSE library. It extends the amazing [CUDArt.jl](https://github.com/JuliaGPU/CUDArt.jl) library to provide three new sparse matrix classes:
    
   -`CudaSparseMatrixCSC`
   -`CudaSparseMatrixCSR`
   -`CudaSparseMatrixBSR`
   -`CudaSparseMatrixHYB`
which implement compressed sparse row/column storage, block CSR, and NVIDIA's hybrid (HYB) COO-ELL format on the GPU. Since Julia's native sparse type is `CSC`, and CUSPARSE's is `CSR`, automatic format conversion is provided, so that when you write
```julia
A = sprand(10,8,0.2)
d_A = CudaSparseMatrixCSR(A)
```
`A` is transformed into `CSR` format and then moved to the GPU. Thus, `d_A` is *not* a transpose of `A`! Similarly, if you have a matrix in dense format on the GPU (in a `CudaArray`), you can simply call `sparse` to turn it into a sparse representation. Right now `sparse` by default turns the matrix it's given into `CSR` format. It takes an optional argument that lets you select `CSC` or `HYB`:

```julia
d_A = CudaArray(rand(10,20))
d_A = sparse(d_A) #now in CSR format

d_B = CudaArray(rand(10,20))
d_B = sparse(d_B,'C') #now in CSC format

d_C = CudaArray(rand(10,20))
d_C = sparse(d_C,'H') #now in HYB format
```
# Current Features

CUSPARSE.jl currently supports a small subset of all the CUSPARSE functionality. What is implemented right now:
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
    - [ ] `bsrsv2_bufferSize`
    - [ ] `bsrsv2_analysis`
    - [ ] `bsrsv2_solve`
    - [ ] `bsrsv2_zeroPivot`
    - [x] `csrsv_analysis`
    - [x] `csrsv_solve`
    - [ ] `csrsv2_bufferSize`
    - [ ] `csrsv2_analysis`
    - [ ] `csrsv2_solve`
    - [ ] `csrsv2_zeroPivot`
    - [x] `hybmv`
    - [x] `hybsv_analysis`
    - [x] `hybsv_solve`
- [ ] Level 3 functions
    - [x] `csrmm`
    - [x] `csrmm2`
    - [x] `csrsm_analysis`
    - [x] `csrsm_solve`
    - [x] `bsrmm`
    - [ ] `bsrsm2_bufferSize`
    - [ ] `bsrsm2_analysis`
    - [ ] `bsrsm2_solve`
    - [ ] `bsrsm2_zeroPivot`
- [ ] Extensions
    - [x] `csrgeam`
    - [x] `csrgemm`
    - [ ] `csrgemm2`
- [ ] Preconditioners
    - [x] `csric0`
    - [ ] `csric02_bufferSize`
    - [ ] `csric02_analysis`
    - [ ] `csric02`
    - [ ] `csric02_zeroPivot`
    - [x] `csrilu0`
    - [ ] `csrilu02_numericBoost`
    - [ ] `csrilu02_bufferSize`
    - [ ] `csrilu02_analysis`
    - [ ] `csrilu02`
    - [ ] `csrilu02_zeroPivot`
    - [ ] `bsric02_bufferSize`
    - [ ] `bsric02_analysis`
    - [ ] `bsric02`
    - [ ] `bsric02_zeroPivot`
    - [ ] `bsrilu02_numericBoost`
    - [ ] `bsrilu02_bufferSize`
    - [ ] `bsrilu02_analysis`
    - [ ] `bsrilu02`
    - [ ] `bsrilu02_zeroPivot`
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

The following type conversions are available:

| From/To: | Dense   | CSR              | CSC              | BSR           | HYB              |
|----------|---------|------------------|------------------|---------------|------------------|
| Dense    | N/A     | sparse(A)        | sparse(A,'C')    | sparse(A,'B') | sparse(A,'H')    |
| CSR      | full(A) | N/A              | switch2csr(A)    | switch2csr(A) | switch2csr(A)    |
| CSC      | full(A) | switch2csc(A)    | N/A              | switch2csc(A) | switch2csc(A)    |
| BSR      | full(A) | switch2bsr(A,bD) | switch2bsr(A,bD) | N/A           | switch2bsr(A,bD) |
| HYB      | full(A) | switch2hyb(A)    | switch2hyb(A)    | switch2hyb(A) | N/A              |

# Working with CUSPARSE.jl

CUSPARSE.jl exports its matrix types, so you don't have to prepend them with anything. To use a CUSPARSE function, just
```julia
using CUSPARSE

### stuff happens here

CUSPARSE.csrmv( #arguments! )
```

# Example
We'll do a simple example of creating two sparse matrices `A`,`B` on the CPU, moving them to the GPU, adding them, and bringing the result back.

```julia
using CUSPARSE
   
# create matrices A,B on the CPU 
A = sprand(10,8,0.3)
B = sprand(8,20,0.4)

# convert A,B to CSR format and
# move them to the GPU - one step
d_A = CudaSparseMatrixCSR(A)
d_B = CudaSparseMatrixCSR(B)

# generate scalar parameters
alpha = rand()
beta  = rand()

# perform alpha * A + beta * B
d_C = CUSPARSE.geam(alpha, d_A, beta, d_B, 'O', 'O')

# bring the result back to the CPU
C = to_host(d_C)
```

Some questions you might have:
- What are the two `'O'`s for?
    - CUSPARSE allows us to use one- or zero-based indexing. Julia uses one-based indexing for arrays, but many other libraries (for instance, C-based libraries) use zero-based. The `'O'`s tell CUSPARSE that our matrices are one-based. If you had a zero-based matrix from an external library, you can tell CUSPARSE using `'Z'`.
- Should we move `alpha` and `beta` to the GPU?
    - We do not have to. CUSPARSE can read in scalar parameters like `alpha` and `beta` from the host (CPU) memory. You can just pass them to the function and CUSPARSE.jl handles telling the CUDA functions where they are for you. If you have an array, like `A` and `B`, you do need to move it to the GPU before CUSPARSE can work on it. Similarly, to see results, if they're in array form, you'll need to move them back to the CPU with `to_host`.
- Since `d_C` is in `CSR` format, is `C` the transpose of what we want?
    - No. CUSPARSE.jl handles the conversion internally so that the final result is in `CSC` format for Julia, and *not* the transpose of the correct answer.

# Contributing

Contributions are very welcome! If you write wrappers for one of the CUSPARSE functions, please include some tests in `test/runtests.jl` for your wrapper. Ideally test each of the types the function you wrap can accept, e.g. `Float32`, `Float64`, and possibly `Complex64`, `Complex128`. You will probably have to add a `ccall` wrapper for your function to `libcusparse.jl` - see there for details.
