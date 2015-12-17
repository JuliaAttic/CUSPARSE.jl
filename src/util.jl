#custom extenstion of CudaArray in CUDArt for sparse vectors/matrices
#using CSC format for interop with Julia's native sparse functionality

import Base: length, size, ndims, eltype, similar, pointer, stride,
    copy, convert, reinterpret, show, summary, copy!, get!, fill!, issym,
    ishermitian, isupper, islower
import Base.LinAlg: BlasFloat, Hermitian, HermOrSym
import CUDArt: device, to_host, free

abstract AbstractCudaSparseArray{Tv,N} <: AbstractSparseArray{Tv,Cint,N}
typealias AbstractCudaSparseVector{Tv} AbstractCudaSparseArray{Tv,1}
typealias AbstractCudaSparseMatrix{Tv} AbstractCudaSparseArray{Tv,2}

type CudaSparseVector{Tv} <: AbstractCudaSparseVector{Tv}
    iPtr::CudaArray{Cint,1}
    nzVal::CudaArray{Tv,1}
    dims::NTuple{2,Int}
    nnz::Cint
    dev::Int
    function CudaSparseVector{Tv}(iPtr::CudaVector{Cint}, nzVal::CudaVector{Tv}, dims::Int, nnz::Cint, dev::Int)
        new(iPtr,nzVal,(dims,1),nnz,dev)
    end
end

type CudaSparseMatrixCSC{Tv} <: AbstractCudaSparseMatrix{Tv}
    colPtr::CudaArray{Cint,1}
    rowVal::CudaArray{Cint,1}
    nzVal::CudaArray{Tv,1}
    dims::NTuple{2,Int}
    nnz::Cint
    dev::Int

    function CudaSparseMatrixCSC{Tv}(colPtr::CudaVector{Cint}, rowVal::CudaVector{Cint}, nzVal::CudaVector{Tv}, dims::NTuple{2,Int}, nnz::Cint, dev::Int)
        new(colPtr,rowVal,nzVal,dims,nnz,dev)
    end
end

type CudaSparseMatrixCSR{Tv} <: AbstractCudaSparseMatrix{Tv}
    rowPtr::CudaArray{Cint,1}
    colVal::CudaArray{Cint,1}
    nzVal::CudaArray{Tv,1}
    dims::NTuple{2,Int}
    nnz::Cint
    dev::Int

    function CudaSparseMatrixCSR{Tv}(rowPtr::CudaVector{Cint}, colVal::CudaVector{Cint}, nzVal::CudaVector{Tv}, dims::NTuple{2,Int}, nnz::Cint, dev::Int)
        new(rowPtr,colVal,nzVal,dims,nnz,dev)
    end
end

type CudaSparseMatrixBSR{Tv} <: AbstractCudaSparseMatrix{Tv}
    rowPtr::CudaArray{Cint,1}
    colVal::CudaArray{Cint,1}
    nzVal::CudaArray{Tv,1}
    dims::NTuple{2,Int}
    blockDim::Cint
    dir::SparseChar
    nnz::Cint
    dev::Int

    function CudaSparseMatrixBSR(rowPtr::CudaVector{Cint}, colVal::CudaVector{Cint}, nzVal::CudaVector{Tv}, dims::NTuple{2,Int},blockDim::Cint, dir::SparseChar, nnz::Cint, dev::Int)
        new(rowPtr,colVal,nzVal,dims,blockDim,dir,nnz,dev)
    end
end

typealias cusparseHybMat_t Ptr{Void}
type CudaSparseMatrixHYB{Tv} <: AbstractCudaSparseMatrix{Tv}
    Mat::cusparseHybMat_t
    dims::NTuple{2,Int}
    nnz::Cint
    dev::Int

    function CudaSparseMatrixHYB(Mat::cusparseHybMat_t, dims::NTuple{2,Int}, nnz::Cint, dev::Int)
        new(Mat,dims,nnz,dev)
    end
end

typealias CompressedSparse{T} Union{CudaSparseMatrixCSC{T},CudaSparseMatrixCSR{T},HermOrSym{T,CudaSparseMatrixCSC{T}},HermOrSym{T,CudaSparseMatrixCSR{T}}}
typealias CudaSparseMatrix{T} Union{CudaSparseMatrixCSC{T},CudaSparseMatrixCSR{T}, CudaSparseMatrixBSR{T}, CudaSparseMatrixHYB{T}}

Hermitian{T}(Mat::CudaSparseMatrix{T}) = Hermitian{T,typeof(Mat)}(Mat,'U')

length(g::CudaSparseVector) = prod(g.dims)
size(g::CudaSparseVector) = g.dims
ndims(g::CudaSparseVector) = 1
length(g::CudaSparseMatrix) = prod(g.dims)
size(g::CudaSparseMatrix) = g.dims
ndims(g::CudaSparseMatrix) = 2

function size{T}(g::CudaSparseVector{T}, d::Integer)
    d >= 1 ? (d <= 1 ? g.dims[d] : 1) : throw(ArgumentError("dimension must be 1, got $d"))
end
function size{T}(g::CudaSparseMatrix{T}, d::Integer)
    d >= 1 ? (d <= 2 ? g.dims[d] : 1) : throw(ArgumentError("dimension must be ≥ 1, got $d"))
end

issym{T}(M::Union{CudaSparseMatrixCSC{T},CudaSparseMatrixCSR{T}})       = false
ishermitian{T}(M::Union{CudaSparseMatrixCSC{T},CudaSparseMatrixCSR{T}}) = false
issym{T}(M::Symmetric{T,CudaSparseMatrixCSC{T}})       = true
ishermitian{T}(M::Hermitian{T,CudaSparseMatrixCSC{T}}) = true

isupper{T}(M::UpperTriangular{T,CudaSparseMatrixCSC{T}}) = true
islower{T}(M::UpperTriangular{T,CudaSparseMatrixCSC{T}}) = false
isupper{T}(M::UpperTriangular{T,CudaSparseMatrixCSR{T}}) = true
islower{T}(M::UpperTriangular{T,CudaSparseMatrixCSR{T}}) = false
isupper{T}(M::UpperTriangular{T,CudaSparseMatrixHYB{T}}) = true
islower{T}(M::UpperTriangular{T,CudaSparseMatrixHYB{T}}) = false
isupper{T}(M::UpperTriangular{T,CudaSparseMatrixBSR{T}}) = true
islower{T}(M::UpperTriangular{T,CudaSparseMatrixBSR{T}}) = false
isupper{T}(M::LowerTriangular{T,CudaSparseMatrixCSC{T}}) = false
islower{T}(M::LowerTriangular{T,CudaSparseMatrixCSC{T}}) = true
isupper{T}(M::LowerTriangular{T,CudaSparseMatrixCSR{T}}) = false
islower{T}(M::LowerTriangular{T,CudaSparseMatrixCSR{T}}) = true
isupper{T}(M::LowerTriangular{T,CudaSparseMatrixHYB{T}}) = false
islower{T}(M::LowerTriangular{T,CudaSparseMatrixHYB{T}}) = true
isupper{T}(M::LowerTriangular{T,CudaSparseMatrixBSR{T}}) = false
islower{T}(M::LowerTriangular{T,CudaSparseMatrixBSR{T}}) = true


eltype{T}(g::CudaSparseMatrix{T}) = T
device(A::CudaSparseMatrix)       = A.dev
device(A::SparseMatrixCSC)        = -1  # for host

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

CudaSparseVector{T<:BlasFloat,Ti<:Integer}(iPtr::Vector{Ti}, nzVal::Vector{T}, dims::Int) = CudaSparseVector{T}(CudaArray(convert(Vector{Cint},iPtr)), CudaArray(nzVal), dims, convert(Cint,length(nzVal)), device())
CudaSparseVector{T<:BlasFloat,Ti<:Integer}(iPtr::CudaArray{Ti}, nzVal::CudaArray{T}, dims::Int) = CudaSparseVector{T}(iPtr, nzVal, dims, convert(Cint,length(nzVal)), device())

CudaSparseMatrixCSC{T<:BlasFloat,Ti<:Integer}(colPtr::Vector{Ti}, rowVal::Vector{Ti}, nzVal::Vector{T}, dims::NTuple{2,Int}) = CudaSparseMatrixCSC{T}(CudaArray(convert(Vector{Cint},colPtr)), CudaArray(convert(Vector{Cint},rowVal)), CudaArray(nzVal), dims, convert(Cint,length(nzVal)), device())
CudaSparseMatrixCSC{T<:BlasFloat,Ti<:Integer}(colPtr::CudaArray{Ti}, rowVal::CudaArray{Ti}, nzVal::CudaArray{T}, dims::NTuple{2,Int}) = CudaSparseMatrixCSC{T}(colPtr, rowVal, nzVal, dims, convert(Cint,length(nzVal)), device())
CudaSparseMatrixCSC{T<:BlasFloat,Ti<:Integer}(colPtr::CudaArray{Ti}, rowVal::CudaArray{Ti}, nzVal::CudaArray{T}, nnz, dims::NTuple{2,Int}) = CudaSparseMatrixCSC{T}(colPtr, rowVal, nzVal, dims, nnz, device())

CudaSparseMatrixCSR{T}(rowPtr::CudaArray, colVal::CudaArray, nzVal::CudaArray{T}, dims::NTuple{2,Int}) = CudaSparseMatrixCSR{T}(rowPtr, colVal, nzVal, dims, convert(Cint,length(nzVal)), device())
CudaSparseMatrixCSR{T}(rowPtr::CudaArray, colVal::CudaArray, nzVal::CudaArray{T}, nnz, dims::NTuple{2,Int}) = CudaSparseMatrixCSR{T}(rowPtr, colVal, nzVal, dims, nnz, device())

CudaSparseMatrixBSR{T}(rowPtr::CudaArray, colVal::CudaArray, nzVal::CudaArray{T}, blockDim, dir, nnz, dims::NTuple{2,Int}) = CudaSparseMatrixBSR{T}(rowPtr, colVal, nzVal, dims, blockDim, dir, nnz, device())

CudaSparseVector(Vec::SparseVector)    = CudaSparseVector(Vec.nzind, Vec.nzval, size(Vec)[1])
CudaSparseMatrixCSC(Vec::SparseVector)    = CudaSparseMatrixCSC([1], Vec.nzind, Vec.nzval, size(Vec))
CudaSparseMatrixCSC(Mat::SparseMatrixCSC) = CudaSparseMatrixCSC(Mat.colptr, Mat.rowval, Mat.nzval, size(Mat))
CudaSparseMatrixCSR(Mat::SparseMatrixCSC) = switch2csr(CudaSparseMatrixCSC(Mat))

similar(Vec::CudaSparseVector) = CudaSparseVector(copy(Vec.iPtr), similar(Vec.nzVal), Vec.dims[1])
similar(Mat::CudaSparseMatrixCSC) = CudaSparseMatrixCSC(copy(Mat.colPtr), copy(Mat.rowVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
similar(Mat::CudaSparseMatrixCSR) = CudaSparseMatrixCSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
similar(Mat::CudaSparseMatrixBSR) = CudaSparseMatrixBSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(Mat.nzVal), Mat.blockDim, Mat.dir, Mat.nnz, Mat.dims)

function copy!(dst::CudaSparseVector, src::CudaSparseVector; stream=null_stream)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Vector size"))
    end
    copy!( dst.iPtr, src.iPtr )
    copy!( dst.nzVal, src.nzVal )
    dst.nnz = src.nnz
    dst
end

function copy!(dst::CudaSparseMatrixCSC, src::CudaSparseMatrixCSC; stream=null_stream)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copy!( dst.colPtr, src.colPtr )
    copy!( dst.rowVal, src.rowVal )
    copy!( dst.nzVal, src.nzVal )
    dst.nnz = src.nnz
    dst
end

function copy!(dst::CudaSparseMatrixCSR, src::CudaSparseMatrixCSR; stream=null_stream)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copy!( dst.rowPtr, src.rowPtr )
    copy!( dst.colVal, src.colVal )
    copy!( dst.nzVal, src.nzVal )
    dst.nnz = src.nnz
    dst
end

function copy!(dst::CudaSparseMatrixBSR, src::CudaSparseMatrixBSR; stream=null_stream)
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

function copy!(dst::CudaSparseMatrixHYB, src::CudaSparseMatrixHYB; stream=null_stream)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    dst.Mat = src.Mat
    dst.nnz = src.nnz
    dst
end

copy(Vec::CudaSparseVector; stream=null_stream) = copy!(similar(Vec),Vec;stream=null_stream)
copy(Mat::CudaSparseMatrixCSC; stream=null_stream) = copy!(similar(Mat),Mat;stream=null_stream)
copy(Mat::CudaSparseMatrixCSR; stream=null_stream) = copy!(similar(Mat),Mat;stream=null_stream)
copy(Mat::CudaSparseMatrixBSR; stream=null_stream) = copy!(similar(Mat),Mat;stream=null_stream)
