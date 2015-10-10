#custom extenstion of CudaArray in CUDArt for sparse vectors/matrices
#using CSC format for interop with Julia's native sparse functionality

import Base: length, size, ndims, eltype, similar, pointer, stride,
    copy, convert, reinterpret, show, summary, copy!, get!, fill!
import CUDArt: device, to_host, free

type CudaSparseMatrixCSC{T}
    colPtr::CudaArray{Cint,1}
    rowVal::CudaArray{Cint,1}
    nzVal::CudaArray{T,1}
    dims::NTuple{2,Int}
    nnz::Cint
    dev::Int

    function CudaSparseMatrixCSC(colPtr::CudaVector{Cint}, rowVal::CudaVector{Cint}, nzVal::CudaVector{T}, dims::NTuple{2,Int}, nnz::Cint, dev::Int)
        new(colPtr,rowVal,nzVal,dims,nnz,dev)
    end
end

type CudaSparseMatrixCSR{T}
    rowPtr::CudaArray{Cint,1}
    colVal::CudaArray{Cint,1}
    nzVal::CudaArray{T,1}
    dims::NTuple{2,Int}
    nnz::Cint
    dev::Int

    function CudaSparseMatrixCSR(rowPtr::CudaVector{Cint}, colVal::CudaVector{Cint}, nzVal::CudaVector{T}, dims::NTuple{2,Int}, nnz::Cint, dev::Int)
        new(rowPtr,colVal,nzVal,dims,nnz,dev)
    end
end

type CudaSparseMatrixBSR{T}
    rowPtr::CudaArray{Cint,1}
    colVal::CudaArray{Cint,1}
    nzVal::CudaArray{T,1}
    dims::NTuple{2,Int}
    blockDim::Cint
    dir::SparseChar
    nnz::Cint
    dev::Int

    function CudaSparseMatrixBSR(rowPtr::CudaVector{Cint}, colVal::CudaVector{Cint}, nzVal::CudaVector{T}, dims::NTuple{2,Int},blockDim::Cint, dir::SparseChar, nnz::Cint, dev::Int)
        new(rowPtr,colVal,nzVal,dims,blockDim,dir,nnz,dev)
    end
end

typealias cusparseHybMat_t Ptr{Void}
type CudaSparseMatrixHYB{T}
    Mat::cusparseHybMat_t
    dims::NTuple{2,Int}
    nnz::Cint
    dev::Int

    function CudaSparseMatrixHYB(Mat::cusparseHybMat_t, dims::NTuple{2,Int}, nnz::Cint, dev::Int)
        new(Mat,dims,nnz,dev)
    end
end

typealias CudaSparseMatrix{T} Union{CudaSparseMatrixCSC{T}, CudaSparseMatrixCSR{T}, CudaSparseMatrixBSR{T}, CudaSparseMatrixHYB{T}}

length(g::CudaSparseMatrix) = prod(g.dims)
size(g::CudaSparseMatrix) = g.dims

function size{T}(g::CudaSparseMatrix{T}, d::Integer)
    d >= 1 ? (d <= 2 ? g.dims[d] : 1) : error("Invalid dim index")
end

eltype{T}(g::CudaSparseMatrix{T}) = T
device(A::CudaSparseMatrix) = A.dev
device(A::SparseMatrixCSC) = -1  # for host

to_host{T}(Mat::CudaSparseMatrixCSC{T}) = SparseMatrixCSC(Mat.dims[1], Mat.dims[2], to_host(Mat.colPtr), to_host(Mat.rowVal), to_host(Mat.nzVal))
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

CudaSparseMatrixCSC(T::Type, colPtr::Vector, rowVal::Vector, nzVal::Vector, dims::NTuple{2,Int}) = CudaSparseMatrixCSC{T}(CudaArray(convert(Vector{Cint},colPtr)), CudaArray(convert(Vector{Cint},rowVal)), CudaArray(nzVal), dims, convert(Cint,length(nzVal)), device())
CudaSparseMatrixCSC(T::Type, colPtr::CudaArray, rowVal::CudaArray, nzVal::CudaArray, dims::NTuple{2,Int}) = CudaSparseMatrixCSC{T}(colPtr, rowVal, nzVal, dims, convert(Cint,length(nzVal)), device())
CudaSparseMatrixCSC(T::Type, colPtr::CudaArray, rowVal::CudaArray, nzVal::CudaArray, nnz, dims::NTuple{2,Int}) = CudaSparseMatrixCSC{T}(colPtr, rowVal, nzVal, dims, nnz, device())

CudaSparseMatrixCSR(T::Type, rowPtr::CudaArray, colVal::CudaArray, nzVal::CudaArray, dims::NTuple{2,Int}) = CudaSparseMatrixCSR{T}(rowPtr, colVal, nzVal, dims, convert(Cint,length(nzVal)), device())
CudaSparseMatrixCSR(T::Type, rowPtr::CudaArray, colVal::CudaArray, nzVal::CudaArray, nnz, dims::NTuple{2,Int}) = CudaSparseMatrixCSR{T}(rowPtr, colVal, nzVal, dims, nnz, device())

CudaSparseMatrixBSR(T::Type, rowPtr::CudaArray, colVal::CudaArray, nzVal::CudaArray, blockDim, dir, nnz, dims::NTuple{2,Int}) = CudaSparseMatrixBSR{T}(rowPtr, colVal, nzVal, dims, blockDim, dir, nnz, device())

CudaSparseMatrixCSC(Mat::SparseMatrixCSC) = CudaSparseMatrixCSC(eltype(Mat), Mat.colptr, Mat.rowval, Mat.nzval, size(Mat))
CudaSparseMatrixCSR(Mat::SparseMatrixCSC) = switch2csr(CudaSparseMatrixCSC(Mat))

similar(Mat::CudaSparseMatrixCSC) = CudaSparseMatrixCSC(eltype(Mat), copy(Mat.colPtr), copy(Mat.rowVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
similar(Mat::CudaSparseMatrixCSR) = CudaSparseMatrixCSR(eltype(Mat), copy(Mat.rowPtr), copy(Mat.colVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
similar(Mat::CudaSparseMatrixBSR) = CudaSparseMatrixBSR(eltype(Mat), copy(Mat.rowPtr), copy(Mat.colVal), similar(Mat.nzVal), Mat.blockDim, Mat.dir, Mat.nnz, Mat.dims)

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

copy(Mat::CudaSparseMatrixCSC; stream=null_stream) = copy!(similar(Mat),Mat;stream=null_stream)
copy(Mat::CudaSparseMatrixCSR; stream=null_stream) = copy!(similar(Mat),Mat;stream=null_stream)
copy(Mat::CudaSparseMatrixBSR; stream=null_stream) = copy!(similar(Mat),Mat;stream=null_stream)
