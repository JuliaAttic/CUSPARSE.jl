#custom extenstion of CuArray in CUDAdrv for sparse vectors/matrices
#using CSC format for interop with Julia's native sparse functionality

import Base: length, size, ndims, eltype, similar, pointer, stride,
    copy, convert, reinterpret, show, summary, copy!, get!, fill!, issymmetric,
    ishermitian, isupper, islower
import Base.LinAlg: BlasFloat, Hermitian, HermOrSym
import CUDAdrv: device
using Compat
export to_host

@compat abstract type AbstractCudaSparseArray{Tv,N} <: AbstractSparseArray{Tv,Cint,N} end
@compat const AbstractCudaSparseVector{Tv} = AbstractCudaSparseArray{Tv,1}
@compat const AbstractCudaSparseMatrix{Tv} = AbstractCudaSparseArray{Tv,2}

"""
Container to hold sparse vectors on the GPU, similar to `SparseVector` in base Julia.
"""
type CudaSparseVector{Tv} <: AbstractCudaSparseVector{Tv}
    iPtr::CuArray{Cint,1}
    nzVal::CuArray{Tv,1}
    dims::NTuple{2,Int}
    nnz::Cint
    
    CudaSparseVector{Tv}(iPtr::CuVector{Cint}, nzVal::CuVector{Tv}, dims::Int, nnz::Cint) where {Tv} = new(iPtr,nzVal,(dims,1),nnz)
end

"""
Container to hold sparse matrices in compressed sparse column (CSC) format on
the GPU, similar to `SparseMatrixCSC` in base Julia.

**Note**: Most CUSPARSE operations work with CSR formatted matrices, rather
than CSC.
"""
type CudaSparseMatrixCSC{Tv} <: AbstractCudaSparseMatrix{Tv}
    colPtr::CuArray{Cint,1}
    rowVal::CuArray{Cint,1}
    nzVal::CuArray{Tv,1}
    dims::NTuple{2,Int}
    nnz::Cint
    

    CudaSparseMatrixCSC{Tv}(colPtr::CuVector{Cint}, rowVal::CuVector{Cint}, nzVal::CuVector{Tv}, dims::NTuple{2,Int}, nnz::Cint) where {Tv} = new(colPtr,rowVal,nzVal,dims,nnz)
end

"""
Container to hold sparse matrices in compressed sparse row (CSR) format on the
GPU.

**Note**: Most CUSPARSE operations work with CSR formatted matrices, rather
than CSC.
"""
type CudaSparseMatrixCSR{Tv} <: AbstractCudaSparseMatrix{Tv}
    rowPtr::CuArray{Cint,1}
    colVal::CuArray{Cint,1}
    nzVal::CuArray{Tv,1}
    dims::NTuple{2,Int}
    nnz::Cint

    CudaSparseMatrixCSR{Tv}(rowPtr::CuVector{Cint}, colVal::CuVector{Cint}, nzVal::CuVector{Tv}, dims::NTuple{2,Int}, nnz::Cint) where {Tv} = new(rowPtr,colVal,nzVal,dims,nnz)
end

"""
Container to hold sparse matrices in block compressed sparse row (BSR) format on
the GPU. BSR format is also used in Intel MKL, and is suited to matrices that are
"block" sparse - rare blocks of non-sparse regions.
"""
type CudaSparseMatrixBSR{Tv} <: AbstractCudaSparseMatrix{Tv}
    rowPtr::CuArray{Cint,1}
    colVal::CuArray{Cint,1}
    nzVal::CuArray{Tv,1}
    dims::NTuple{2,Int}
    blockDim::Cint
    dir::SparseChar
    nnz::Cint
    

    CudaSparseMatrixBSR{Tv}(rowPtr::CuVector{Cint}, colVal::CuVector{Cint}, nzVal::CuVector{Tv}, dims::NTuple{2,Int},blockDim::Cint, dir::SparseChar, nnz::Cint) where {Tv} = new(rowPtr,colVal,nzVal,dims,blockDim,dir,nnz)
end

"""
Container to hold sparse matrices in NVIDIA's hybrid (HYB) format on the GPU.
HYB format is an opaque struct, which can be converted to/from using
CUSPARSE routines.
"""
const cusparseHybMat_t = Ptr{Void}
type CudaSparseMatrixHYB{Tv} <: AbstractCudaSparseMatrix{Tv}
    Mat::cusparseHybMat_t
    dims::NTuple{2,Int}
    nnz::Cint
    

    CudaSparseMatrixHYB{Tv}(Mat::cusparseHybMat_t, dims::NTuple{2,Int}, nnz::Cint) where {Tv} = new(Mat,dims,nnz)
end

"""
Utility union type of [`CudaSparseMatrixCSC`](@ref), [`CudaSparseMatrixCSR`](@ref),
and `Hermitian` and `Symmetric` versions of these two containers. A function accepting
this type can make use of performance improvements by only indexing one triangle of the
matrix if it is guaranteed to be hermitian/symmetric.
"""
@compat const CompressedSparse{T} = Union{CudaSparseMatrixCSC{T},CudaSparseMatrixCSR{T},HermOrSym{T,CudaSparseMatrixCSC{T}},HermOrSym{T,CudaSparseMatrixCSR{T}}}

"""
Utility union type of [`CudaSparseMatrixCSC`](@ref), [`CudaSparseMatrixCSR`](@ref),
[`CudaSparseMatrixBSR`](@ref), and [`CudaSparseMatrixHYB`](@ref).
"""
@compat const CudaSparseMatrix{T} = Union{CudaSparseMatrixCSC{T},CudaSparseMatrixCSR{T}, CudaSparseMatrixBSR{T}, CudaSparseMatrixHYB{T}}

Hermitian{T}(Mat::CudaSparseMatrix{T}) = Hermitian{T,typeof(Mat)}(Mat,'U')

length(g::CudaSparseVector) = prod(g.dims)
size(g::CudaSparseVector) = g.dims
ndims(g::CudaSparseVector) = 1
length(g::CudaSparseMatrix) = prod(g.dims)
size(g::CudaSparseMatrix) = g.dims
ndims(g::CudaSparseMatrix) = 2

function size{T}(g::CudaSparseVector{T}, d::Integer)
    if d == 1
        return g.dims[d]
    elseif d > 1
        return 1
    else
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
end

function size{T}(g::CudaSparseMatrix{T}, d::Integer)
    if d in [1, 2]
        return g.dims[d]
    elseif d > 1
        return 1
    else
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
end

issymmetric{T}(M::Union{CudaSparseMatrixCSC{T},CudaSparseMatrixCSR{T}}) = false
ishermitian{T}(M::Union{CudaSparseMatrixCSC{T},CudaSparseMatrixCSR{T}}) = false
issymmetric{T}(M::Symmetric{T,CudaSparseMatrixCSC{T}}) = true
ishermitian{T}(M::Hermitian{T,CudaSparseMatrixCSC{T}}) = true

for mat_type in [:CudaSparseMatrixCSC, :CudaSparseMatrixCSR, :CudaSparseMatrixBSR, :CudaSparseMatrixHYB]
    @eval begin
        isupper{T}(M::UpperTriangular{T,$mat_type{T}}) = true
        islower{T}(M::UpperTriangular{T,$mat_type{T}}) = false
    end
end
eltype{T}(g::CudaSparseMatrix{T}) = T
device(A::CudaSparseMatrix)       = A.dev
device(A::SparseMatrixCSC)        = -1  # for host

to_host{T}(g::CUDAdrv.CuArray{T}) = copy!(Array{T}(size(g)), g)

function to_host{T}(Vec::CudaSparseVector{T})
    SparseVector(Vec.dims[1], to_host(Vec.iPtr), to_host(Vec.nzVal))
end

function to_host{T}(Mat::CudaSparseMatrixCSC{T})
    SparseMatrixCSC(Mat.dims[1], Mat.dims[2], to_host(Mat.colPtr), to_host(Mat.rowVal), to_host(Mat.nzVal))

end
function to_host{T}(Mat::CudaSparseMatrixCSR{T})
    rowPtr = to_host(Mat.rowPtr)
    colVal = to_host(Mat.colVal)
    nzVal = to_host(Mat.nzVal)
    #construct Is
    I = similar(colVal)
    counter = 1
    for row = 1 : size(Mat)[1], k = rowPtr[row] : (rowPtr[row+1]-1)
        I[counter] = row
        counter += 1
    end
    return sparse(I,colVal,nzVal,Mat.dims[1],Mat.dims[2])
end

summary(g::CudaSparseMatrix) = string(g)
summary(g::CudaSparseVector) = string(g)

CudaSparseVector{T<:BlasFloat,Ti<:Integer}(iPtr::Vector{Ti}, nzVal::Vector{T}, dims::Int) = CudaSparseVector{T}(CuArray(convert(Vector{Cint},iPtr)), CuArray(nzVal), dims, convert(Cint,length(nzVal)))
CudaSparseVector{T<:BlasFloat,Ti<:Integer}(iPtr::CuArray{Ti}, nzVal::CuArray{T}, dims::Int) = CudaSparseVector{T}(iPtr, nzVal, dims, convert(Cint,length(nzVal)))

CudaSparseMatrixCSC{T<:BlasFloat,Ti<:Integer}(colPtr::Vector{Ti}, rowVal::Vector{Ti}, nzVal::Vector{T}, dims::NTuple{2,Int}) = CudaSparseMatrixCSC{T}(CuArray(convert(Vector{Cint},colPtr)), CuArray(convert(Vector{Cint},rowVal)), CuArray(nzVal), dims, convert(Cint,length(nzVal)))
CudaSparseMatrixCSC{T<:BlasFloat,Ti<:Integer}(colPtr::CuArray{Ti}, rowVal::CuArray{Ti}, nzVal::CuArray{T}, dims::NTuple{2,Int}) = CudaSparseMatrixCSC{T}(colPtr, rowVal, nzVal, dims, convert(Cint,length(nzVal)))
CudaSparseMatrixCSC{T<:BlasFloat,Ti<:Integer}(colPtr::CuArray{Ti}, rowVal::CuArray{Ti}, nzVal::CuArray{T}, nnz, dims::NTuple{2,Int}) = CudaSparseMatrixCSC{T}(colPtr, rowVal, nzVal, dims, nnz)

CudaSparseMatrixCSR{T}(rowPtr::CuArray, colVal::CuArray, nzVal::CuArray{T}, dims::NTuple{2,Int}) = CudaSparseMatrixCSR{T}(rowPtr, colVal, nzVal, dims, convert(Cint,length(nzVal)))
CudaSparseMatrixCSR{T}(rowPtr::CuArray, colVal::CuArray, nzVal::CuArray{T}, nnz, dims::NTuple{2,Int}) = CudaSparseMatrixCSR{T}(rowPtr, colVal, nzVal, dims, nnz)

CudaSparseMatrixBSR{T}(rowPtr::CuArray, colVal::CuArray, nzVal::CuArray{T}, blockDim, dir, nnz, dims::NTuple{2,Int}) = CudaSparseMatrixBSR{T}(rowPtr, colVal, nzVal, dims, blockDim, dir, nnz)

CudaSparseVector(Vec::SparseVector)    = CudaSparseVector(Vec.nzind, Vec.nzval, size(Vec)[1])
CudaSparseMatrixCSC(Vec::SparseVector)    = CudaSparseMatrixCSC([1], Vec.nzind, Vec.nzval, size(Vec))
CudaSparseVector(Mat::SparseMatrixCSC) = size(Mat,2) == 1 ? CudaSparseVector(Mat.rowval, Mat.nzval, size(Mat)[1]) : throw(ArgumentError())
CudaSparseMatrixCSC(Mat::SparseMatrixCSC) = CudaSparseMatrixCSC(Mat.colptr, Mat.rowval, Mat.nzval, size(Mat))
CudaSparseMatrixCSR(Mat::SparseMatrixCSC) = switch2csr(CudaSparseMatrixCSC(Mat))

similar(Vec::CudaSparseVector) = CudaSparseVector(copy(Vec.iPtr), similar(Vec.nzVal), Vec.dims[1])
similar(Mat::CudaSparseMatrixCSC) = CudaSparseMatrixCSC(copy(Mat.colPtr), copy(Mat.rowVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
similar(Mat::CudaSparseMatrixCSR) = CudaSparseMatrixCSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
similar(Mat::CudaSparseMatrixBSR) = CudaSparseMatrixBSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(Mat.nzVal), Mat.blockDim, Mat.dir, Mat.nnz, Mat.dims)

#TODO: Ask what streams were?
function copy!(dst::CudaSparseVector, src::CudaSparseVector)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Vector size"))
    end
    copy!( dst.iPtr, src.iPtr )
    copy!( dst.nzVal, src.nzVal )
    dst.nnz = src.nnz
    dst
end

function copy!(dst::CudaSparseMatrixCSC, src::CudaSparseMatrixCSC)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copy!( dst.colPtr, src.colPtr )
    copy!( dst.rowVal, src.rowVal )
    copy!( dst.nzVal, src.nzVal )
    dst.nnz = src.nnz
    dst
end

function copy!(dst::CudaSparseMatrixCSR, src::CudaSparseMatrixCSR)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copy!( dst.rowPtr, src.rowPtr )
    copy!( dst.colVal, src.colVal )
    copy!( dst.nzVal, src.nzVal )
    dst.nnz = src.nnz
    dst
end

function copy!(dst::CudaSparseMatrixBSR, src::CudaSparseMatrixBSR)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copy!( dst.rowPtr, src.rowPtr )
    copy!( dst.colVal, src.colVal )
    copy!( dst.nzVal, src.nzVal )
    dst.dir = src.dir
    dst.nnz = src.nnz
    dst
end

function copy!(dst::CudaSparseMatrixHYB, src::CudaSparseMatrixHYB)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    dst.Mat = src.Mat
    dst.nnz = src.nnz
    dst
end

copy(Vec::CudaSparseVector) = copy!(similar(Vec),Vec)
copy(Mat::CudaSparseMatrixCSC) = copy!(similar(Mat),Mat)
copy(Mat::CudaSparseMatrixCSR) = copy!(similar(Mat),Mat)
copy(Mat::CudaSparseMatrixBSR) = copy!(similar(Mat),Mat)
