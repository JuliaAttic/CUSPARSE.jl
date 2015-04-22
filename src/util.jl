#custom extenstion of CudaArray in CUDArt for sparse vectors/matrices
#using CSC format for interop with Julia's native sparse functionality

import Base: length, size, ndims, eltype, similar, pointer, stride,
    copy, convert, reinterpret, show, summary,
    copy!, get!, fill!
import CUDArt: device, to_host

type CudaSparseMatrix{T}
    colPtr::CudaArray{Cint,1}
    rowVal::CudaArray{Cint,1}
    nzVal::CudaArray{T,1}
    dims::NTuple{2,Int}
    nnz::Cint
    dev::Int

    function CudaSparseMatrix(colPtr::CudaVector{Cint}, rowVal::CudaVector{Cint}, nzVal::CudaVector{T}, dims::NTuple{2,Int}, nnz::Cint, dev::Int)
        new(colPtr,rowVal,nzVal,dims,nnz,dev)
    end
end

length(g::CudaSparseMatrix) = prod(g.dims)
size(g::CudaSparseMatrix) = g.dims

function size{T}(g::CudaSparseMatrix{T}, d::Integer)
    d >= 1 ? (d <= 2 ? g.dims[d] : 1) : error("Invalid dim index")
end

eltype{T}(g::CudaSparseMatrix{T}) = T
device(A::CudaSparseMatrix) = A.dev
device(A::SparseMatrixCSC) = -1  # for host

colpointer(g::CudaSparseMatrix) = g.colPtr
nonzeros(g::CudaSparseMatrix)   = g.nzVal
rowvals(g::CudaSparseMatrix)    = g.rowVal
pointers(g::CudaSparseMatrix)   = (colpointer(g),rowvals(g),nonzeros(g))

to_host{T}(Mat::CudaSparseMatrix{T}) = SparseMatrixCSC(Mat.dims[1], Mat.dims[2], to_host(Mat.colPtr), to_host(Mat.rowVal), to_host(Mat.nzVal))

summary(g::CudaSparseMatrix) = string(g)

CudaSparseMatrix(T::Type, colPtr::Vector, rowVal::Vector, nzVal::Vector, dims::NTuple{2,Int}) = CudaSparseMatrix{T}(CudaArray(convert(Vector{Cint},colPtr)), CudaArray(convert(Vector{Cint},rowVal)), CudaArray(nzVal), dims, convert(Cint,length(nzVal)), device())
CudaSparseMatrix(T::Type, colPtr::Vector, rowVal::Vector, nzVal::Vector, nnz, dims::NTuple{2,Int}) = CudaSparseMatrix{T}(CudaArray(convert(Vector{Cint},colPtr)), CudaArray(convert(Vector{Cint},rowVal)), CudaArray(nzVal), dims, convert(Cint,nnz), device())
CudaSparseMatrix(T::Type, colPtr::CudaArray, rowVal::CudaArray, nzVal::CudaArray, dims::NTuple{2,Int}) = CudaSparseMatrix{T}(colPtr, rowVal, nzVal, dims, convert(Cint,length(nzVal)), device())
CudaSparseMatrix(T::Type, colPtr::CudaArray, rowVal::CudaArray, nzVal::CudaArray, nnz, dims::NTuple{2,Int}) = CudaSparseMatrix{T}(colPtr, rowVal, nzVal, dims, nnz, device())


CudaSparseMatrix(Mat::SparseMatrixCSC) = CudaSparseMatrix(eltype(Mat), Mat.colptr, Mat.rowval, Mat.nzval, size(Mat))

similar(Mat::CudaSparseMatrix) = CudaSparseMatrix(eltype(Mat), Mat.colPtr, Mat.rowVal, CudaArray(zeros(eltype(Mat),Mat.nnz)), Mat.nnz, Mat.dims)

function copy!(dst::CudaSparseMatrix, src::CudaSparseMatrix; stream=null_stream)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copy!( dst.colPtr, src.colPtr ) 
    copy!( dst.rowVal, src.rowVal ) 
    copy!( dst.nzVal, src.nzVal )
    dst.nnz = src.nnz
    dst
end

copy(Mat::CudaSparseMatrix; stream=null_stream) = copy!(similar(Mat),Mat;stream=null_stream)
