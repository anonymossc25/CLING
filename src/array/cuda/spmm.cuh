/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/spmm.cuh
 * @brief SPMM CUDA kernel function header.
 */
#ifndef DGL_ARRAY_CUDA_SPMM_CUH_
#define DGL_ARRAY_CUDA_SPMM_CUH_
#include <curand_kernel.h>

#include <dgl/bcast.h>

#include <limits>
#include <iostream>

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"
#include "atomic.cuh"
#include "bf16.cuh"
#include "fp16.cuh"
#include "macro.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.hpp>  // for __nv_bfloat162float

namespace dgl {

using namespace cuda;
using namespace std;

namespace aten {

/**
 * @brief Determine whether cusparse SpMM function is applicable.
 */
template <typename DType, typename IdType>
inline bool cusparse_available(bool more_nnz_than_matrix_size) {
#if CUDART_VERSION < 11000
  if (std::is_same<IdType, int>::value &&
      (std::is_same<DType, float>::value || std::is_same<DType, double>::value))
    return true;
  return false;
#else
  if (std::is_same<DType, __half>::value ||
      std::is_same<DType, __nv_bfloat16>::value)
    return false;  // cusparse's SpMM on fp16 is slow, temporally disabled.
  // If the CSR matrix has more NNZ than matrix size, we should not use
  // cuSPARSE 11.1.
  return !more_nnz_than_matrix_size;
#endif
}

namespace {

/** @brief Call cuBLAS geam API for transpose operation for float and double. */
template <typename DType>
cublasStatus_t Xgeam(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const DType* alpha, const DType* A, int lda,
    const DType* beta, const DType* B, int ldb, DType* C, int ldc) {
  LOG(FATAL) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t Xgeam<__half>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const __half* alpha, const __half* A, int lda,
    const __half* beta, const __half* B, int ldb, __half* C, int ldc) {
  // TODO(ndickson): There is no cublasHgeam, so a different
  // implementation would be required.
  LOG(FATAL) << "Xgeam does not support dtype half (FP16)";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

#if BF16_ENABLED
template <>
cublasStatus_t Xgeam<__nv_bfloat16>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const __nv_bfloat16* alpha, const __nv_bfloat16* A, int lda,
    const __nv_bfloat16* beta, const __nv_bfloat16* B, int ldb,
    __nv_bfloat16* C, int ldc) {
  // TODO(ndickson): There is no cublasHgeam, so a different
  // implementation would be required.
  LOG(FATAL) << "Xgeam does not support dtype bfloat16 (BF16)";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}
#endif  // BF16_ENABLED

template <>
cublasStatus_t Xgeam<float>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb, float* C, int ldc) {
  return cublasSgeam(
      handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

template <>
cublasStatus_t Xgeam<double>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const double* alpha, const double* A, int lda,
    const double* beta, const double* B, int ldb, double* C, int ldc) {
  return cublasDgeam(
      handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

/**
 * @brief Transpose operator kernel implementation.
 * @note not efficient but it's not a bottleneck, used for float16 dtype.
 */
template <typename DType>
__global__ void _TransposeKernel(
    const DType* __restrict__ in, DType* __restrict__ out, int n, int m) {
  int i = blockIdx.x;
  for (int j = threadIdx.x; j < m; j += blockDim.x)
    out[i * m + j] = in[j * n + i];
}

/**
 * @brief Tranpose the input matrix.
 * @param row number of rows of input matrix.
 * @param col number of columns of input matrix.
 */
template <typename DType>
void _Transpose(const DType* in, DType* out, int row, int col) {
  DType alpha = 1., beta = 0.;
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  if (!thr_entry->cublas_handle)
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, stream));
  CUBLAS_CALL(Xgeam<DType>(
      thr_entry->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, row, col, &alpha, in,
      col, &beta, nullptr, row, out, row));
}

/**
 * @brief Tranpose the input matrix for data type half.
 * @note cuBLAS has no geam API for half data type, fallback to our kernel.
 */
template <>
void _Transpose<__half>(const __half* in, __half* out, int row, int col) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int nt = FindNumThreads(row);
  int nb = col;
  CUDA_KERNEL_CALL(_TransposeKernel, nb, nt, 0, stream, in, out, col, row);
}

#if BF16_ENABLED
/**
 * @brief Tranpose the input matrix for data type half.
 * @note cuBLAS has no geam API for bf16 data type, fallback to our kernel.
 */
template <>
void _Transpose<__nv_bfloat16>(
    const __nv_bfloat16* in, __nv_bfloat16* out, int row, int col) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int nt = FindNumThreads(row);
  int nb = col;
  CUDA_KERNEL_CALL(_TransposeKernel, nb, nt, 0, stream, in, out, col, row);
}
#endif  // BF16_ENABLED

#if CUDART_VERSION < 11000
template <typename DType>
cusparseStatus_t Xcsrmm2(
    cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const DType* alpha, const cusparseMatDescr_t descrA, const DType* csrValA,
    const int* csrRowPtrA, const int* csrColIndA, const DType* B, int ldb,
    const DType* beta, DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <>
cusparseStatus_t Xcsrmm2<float>(
    cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const float* alpha, const cusparseMatDescr_t descrA, const float* csrValA,
    const int* csrRowPtrA, const int* csrColIndA, const float* B, int ldb,
    const float* beta, float* C, int ldc) {
  return cusparseScsrmm2(
      handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
      csrColIndA, B, ldb, beta, C, ldc);
}

template <>
cusparseStatus_t Xcsrmm2<double>(
    cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const double* alpha, const cusparseMatDescr_t descrA, const double* csrValA,
    const int* csrRowPtrA, const int* csrColIndA, const double* B, int ldb,
    const double* beta, double* C, int ldc) {
  return cusparseDcsrmm2(
      handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
      csrColIndA, B, ldb, beta, C, ldc);
}
#endif

/** Cusparse implementation of SpMM on Csr format. */
template <typename DType, typename IdType>
void CusparseCsrmm2(
    const DGLContext& ctx, const CSRMatrix& csr, const DType* B_data,
    const DType* A_data, DType* C_data, int x_length) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix
  // for node feature tensor. However, since cusparse only supports
  // column-major, while our tensor is stored in row-major, the actual
  // computation is: C = trans(A x trans(B)). Currently, we use cublasXgeam to
  // implement transposition and allocate intermediate workspace memory for
  // this.
  // Total size of the array (in bytes)
 //  std::cout << "Total size of array (in bytes): " << sizeof(A_data) << std::endl;
 //
 //    // Size of one row (in bytes)
 //  std::cout << "Size of one row (in bytes): " << sizeof(A_data[0]) << std::endl;
 //
 //    // Number of rows
 //  std::cout << "Number of rows: " << sizeof(A_data) / sizeof(A_data[0]) << std::endl;
 // // size of dtype
 //  std::cout << "Size of DType: " << sizeof(DType) << std::endl;

  // size of first element
  //std::cout <<"size of first element " << A_data[0] << std::endl;
  // printf("A_data: ");
  // for(int i=0;i<2;i++)
  //   for(int j=0;j<1;j++)
  //     std::cout<<A_data[i+j]<<' ';
  //   std::cout<<"\n";

    // Number of columns
  //std::cout << "Number of columns: " << sizeof(A_data[0]) / sizeof(A_data[0][0]) << std::endl;
  // std::cout << "Size of DType: " << sizeof(DType) << " bytes\n";
  // std::cout << "Size of float: " << sizeof(float) << " bytes\n";
  // std::cout << "Size of double: " << sizeof(double) << " bytes\n";
  // std::cout << "Size of half: " << sizeof(__half) << " bytes\n";
  const int m = csr.num_rows;
  const int n = x_length;
  const int k = csr.num_cols;
  const int nnz = csr.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 0.0;
  static float spmm_time = 0.0;
  //printf("Inside spmm.cuh\n");
  //printf("\n CusparseCsrmm2 line 214");
  // device
  //printf("m=%d,n=%d,k=%d,nnz=%d\n",m,n,k,nnz);
  //printf("csr num_rows %d, num_x_lenght %d, nnz %d, num_cols %d\n", m, n, nnz,k);
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, stream));
  //correct way to print A_data
  /*printf("Printing A data\n");
  float* h_A_data = new float[nnz];
    cudaMemcpy(h_A_data, A_data, nnz * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nnz; ++i) {
        std::cout << "A_data[" << i << "] = " << h_A_data[i] << std::endl;
    }
    delete[] h_A_data;
  printf("\n");
  */
  // all one data array
    // Print elements of A_data
  //std::cout << "m: " << m << "k:" << k << std::endl;
  // Assuming A_data is a 2D array (m x k)
  // Assuming A_data is a device pointer (allocated on GPU)
  //copy A_data o host and printing
//  DType* host_A_data = nullptr;
// DType* host_A_data;
//
//
// cudaMallocHost(&host_A_data, sizeof(DType) * m * k); // Allocate pinned memory on host
//
// // Copy the data from device to host
// cudaMemcpy(host_A_data, A_data, sizeof(DType) * m * k, cudaMemcpyDeviceToHost);
//
// // Now print the values from host_A_data
// for (int i = 0; i < m; ++i) {
//     for (int j = 0; j < k; ++j) {
//         std::cout << static_cast<float>(host_A_data[i * k + j]) << " ";
//     }
//     std::cout << std::endl;
// }
//
// // Free host memory
// cudaFreeHost(host_A_data);

  // std::cout << "Elements of A_data:" << std::endl;
  // for (int i = 0; i < m; ++i) {
  //   for (int j = 0; j < k; ++j) {
  //     std::cout << static_cast<float>(A_data[i * k + j]) << " ";
  //     //std::cout << __nv_bfloat162float(A_data[i * k + j]) << " ";
  //     //std::cout << __half2float(A_data[i * k + j]) << " ";
  //     //std::cout << A_data[i * k + j] << " "; // Accessing A_data[i][j]
  //   }
  //   std::cout << std::endl;
  // }

  DType* valptr = nullptr;
  if (!A_data) {
    valptr =
        static_cast<DType*>(device->AllocWorkspace(ctx, nnz * sizeof(DType)));
    _Fill(valptr, nnz, static_cast<DType>(1.));
    //printf("Allocating value to A data\n");
  }
   //printf("Printing valptr\n");
   // for (int i = 0; i < nnz; ++i) {
   //      std::cout << "valptr[" << i << "] = " << valptr[i] << std::endl;
   //    }
 /* 
  printf("Printing valptr\n");
  DType* h_valptr = new DType[nnz];
    cudaMemcpy(h_valptr, valptr, nnz * sizeof(DType), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nnz; ++i) {
        std::cout << h_valptr[i];
    }
    delete[] h_valptr;
  printf("\n");
  */
#if CUDART_VERSION >= 11000
  //printf("1st part\n");
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  constexpr auto dtype = cuda_dtype<DType>::value;
  constexpr auto idtype = cusparse_idtype<IdType>::value;
  CUSPARSE_CALL(cusparseCreateCsr(
      &matA, m, k, nnz, static_cast<IdType*>(csr.indptr->data),
      static_cast<IdType*>(csr.indices->data),
      const_cast<DType*>(valptr ? valptr : A_data), idtype, idtype,
      CUSPARSE_INDEX_BASE_ZERO, dtype));
  CUSPARSE_CALL(cusparseCreateDnMat(
      &matB, k, n, n, const_cast<DType*>(B_data), dtype, CUSPARSE_ORDER_ROW));
  CUSPARSE_CALL(
      cusparseCreateDnMat(&matC, m, n, n, C_data, dtype, CUSPARSE_ORDER_ROW));

  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t workspace_size;
  CUSPARSE_CALL(cusparseSpMM_bufferSize(
      thr_entry->cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, CUSPARSE_SPMM_CSR_ALG2, &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  

  CUSPARSE_CALL(cusparseSpMM(
      thr_entry->cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, CUSPARSE_SPMM_CSR_ALG2, workspace));
  device->FreeWorkspace(ctx, workspace);
  
  cudaEventRecord(stop); // Record stop event after your CUDA operation
  cudaEventSynchronize(stop); // Synchronize on stop event to ensure it has completed
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  spmm_time += milliseconds/1000;
  printf("spmm time %.6f\n",spmm_time);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  CUSPARSE_CALL(cusparseDestroySpMat(matA));
  CUSPARSE_CALL(cusparseDestroyDnMat(matB));
  CUSPARSE_CALL(cusparseDestroyDnMat(matC));
#else
  // allocate matrix for temporary transposed output
  printf("2nd part\n");
  DType* trans_out =
      static_cast<DType*>(device->AllocWorkspace(ctx, m * n * sizeof(DType)));

  cusparseMatDescr_t descr;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
  CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CALL(Xcsrmm2<DType>(
      thr_entry->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, m, n, k, nnz, &alpha, descr,
      (valptr) ? valptr : A_data, static_cast<int32_t*>(csr.indptr->data),
      static_cast<int32_t*>(csr.indices->data), B_data, n, &beta, trans_out,
      m));
  CUSPARSE_CALL(cusparseDestroyMatDescr(descr));
  // transpose the output matrix
  _Transpose(trans_out, C_data, n, m);
  device->FreeWorkspace(ctx, trans_out);
#endif
  if (valptr) device->FreeWorkspace(ctx, valptr);
}

/** Cusparse implementation of SpMM on Csr format. */
template <typename DType, typename IdType>
void CusparseCsrmm2Hetero(
    const DGLContext& ctx, const CSRMatrix& csr, const DType* B_data,
    const DType* A_data, DType* C_data, int64_t x_length,
    cudaStream_t strm_id) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix
  // for node feature tensor. However, since cusparse only supports
  // column-major, while our tensor is stored in row-major, the actual
  // computation is: C = trans(A x trans(B)). Currently, we use cublasXgeam to
  // implement transposition and allocate intermediate workspace memory for
  // this.
  int int_maxlimit = std::numeric_limits<int>::max();
  CHECK_GE(int_maxlimit, (csr.num_rows));
  CHECK_GE(int_maxlimit, csr.num_cols);
  CHECK_GE(int_maxlimit, csr.indices->shape[0]);
  const int m = csr.num_rows;
  const int n = x_length;
  const int k = csr.num_cols;
  const int nnz = csr.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 1.0;
  printf("\n CusparseCsrmm2Hetro line 307");
  // device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, strm_id));
  // all one data array
  DType* valptr = nullptr;
  if (!A_data) {
    valptr =
        static_cast<DType*>(device->AllocWorkspace(ctx, nnz * sizeof(DType)));
    _Fill(valptr, nnz, static_cast<DType>(1.));
  }
#if CUDART_VERSION >= 11000
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  constexpr auto dtype = cuda_dtype<DType>::value;
  constexpr auto idtype = cusparse_idtype<IdType>::value;
  CUSPARSE_CALL(cusparseCreateCsr(
      &matA, m, k, nnz, static_cast<IdType*>(csr.indptr->data),
      static_cast<IdType*>(csr.indices->data),
      const_cast<DType*>(valptr ? valptr : A_data), idtype, idtype,
      CUSPARSE_INDEX_BASE_ZERO, dtype));
  CUSPARSE_CALL(cusparseCreateDnMat(
      &matB, k, n, n, const_cast<DType*>(B_data), dtype, CUSPARSE_ORDER_ROW));
  CUSPARSE_CALL(
      cusparseCreateDnMat(&matC, m, n, n, C_data, dtype, CUSPARSE_ORDER_ROW));

  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t workspace_size;
  CUSPARSE_CALL(cusparseSpMM_bufferSize(
      thr_entry->cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, CUSPARSE_SPMM_CSR_ALG2, &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  CUSPARSE_CALL(cusparseSpMM(
      thr_entry->cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, CUSPARSE_SPMM_CSR_ALG2, workspace));
  device->FreeWorkspace(ctx, workspace);

  CUSPARSE_CALL(cusparseDestroySpMat(matA));
  CUSPARSE_CALL(cusparseDestroyDnMat(matB));
  CUSPARSE_CALL(cusparseDestroyDnMat(matC));
#else
  cusparseMatDescr_t descr;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
  CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  CHECK_EQ(sizeof(IdType), sizeof(int32_t));
  CUSPARSE_CALL(Xcsrmm2<DType>(
      thr_entry->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, m, n, k, nnz, &alpha, descr,
      (valptr) ? valptr : A_data, static_cast<int32_t*>(csr.indptr->data),
      static_cast<int32_t*>(csr.indices->data), B_data, n, &beta, C_data, m));
  CUSPARSE_CALL(cusparseDestroyMatDescr(descr));
#endif
  if (valptr) device->FreeWorkspace(ctx, valptr);
}

}  // namespace

#define SWITCH_OP(op, Op, ...)                                  \
  do {                                                          \
    if ((op) == "add") {                                        \
      typedef cuda::binary::Add<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "sub") {                                 \
      typedef cuda::binary::Sub<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "mul") {                                 \
      typedef cuda::binary::Mul<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "div") {                                 \
      typedef cuda::binary::Div<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "copy_lhs") {                            \
      typedef cuda::binary::CopyLhs<DType> Op;                  \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "copy_rhs") {                            \
      typedef cuda::binary::CopyRhs<DType> Op;                  \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op; \
    }                                                           \
  } while (0)

namespace cuda {

/**
 * @brief CUDA kernel of g-SpMM on Coo format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension. To avoid possible data hazards, it uses
 * atomic operators for reduction.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCooKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ row, const Idx* __restrict__ col,
    const Idx* __restrict__ edge_map, int64_t N, int64_t M, int64_t E,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
	printf("SpMMCooKerne\n");
  // SPMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len) : nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
    DType* outoff = out + dst * out_len;
    while (tx < out_len) {
      const int64_t lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int64_t rhs_add = UseBcast ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      Idx* arguoff = nullptr;  // arguoff is not used in SpMMCoo.
      Idx* argeoff = nullptr;  // argeoff is not used in SpMMCoo.
      ReduceOp::Call(outoff + tx, arguoff, argeoff, val, src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel to compute argu and arge in g-SpMM on Coo format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void ArgSpMMCooKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ row, const Idx* __restrict__ col,
    const Idx* __restrict__ edge_map, int64_t N, int64_t M, int64_t E,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
  // SPMM with COO arg max/min.
	printf("ArgSpMMCooKernel\n");
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len) : nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
    const DType* outoff = out + dst * out_len;
    Idx* arguoff = BinaryOp::use_lhs ? (arg_u + dst * out_len) : nullptr;
    Idx* argeoff = BinaryOp::use_rhs ? (arg_e + dst * out_len) : nullptr;
    while (tx < out_len) {
      int64_t lhs_add = UseBcast ? ubcast_off[tx] : tx;
      int64_t rhs_add = UseBcast ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      ReduceOp::CallArg(tx, arguoff, argeoff, val, outoff[tx], src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel of g-SpMM on Csr format.
 * @note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCsrKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ indptr, const Idx* __restrict__ indices,
    const Idx* __restrict__ edge_map, int64_t num_rows, int64_t num_cols,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
    printf("SpMMCsrKernel\n");	
  // SPMM with CSR.
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.x;
  const int stride_x = blockDim.x * gridDim.y;
  while (ty < num_rows) {
    int tx = blockIdx.y * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      typename accum_dtype<DType>::type local_accum = ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff =
            BinaryOp::use_lhs ? (ufeat + cid * ufeat_len) : nullptr;
        const DType* eoff =
            BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
        DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
      }
      // The use of += is to compute cross-type reducing on heterogeneous graph
      // when reduce op is `sum`.
      //     C = SpMM(SpA, B) + C
      // Separate kernel `SpMMCmpCsrHeteroKernel` is used for max- and
      // min-reducer. It does not affect the output on homogeneous graph as
      // `out` is initialized to zero.
      out[ty * out_len + tx] += static_cast<DType>(local_accum);
      if (ReduceOp::require_arg && BinaryOp::use_lhs)
        arg_u[ty * out_len + tx] = local_argu;
      if (ReduceOp::require_arg && BinaryOp::use_rhs)
        arg_e[ty * out_len + tx] = local_arge;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel of SpMM-Min/Max on Csr format.
 * @note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCmpCsrHeteroKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    Idx* __restrict__ arg_u_ntype, Idx* __restrict__ arg_e_etype,
    const Idx* __restrict__ indptr, const Idx* __restrict__ indices,
    const Idx* __restrict__ edge_map, int64_t num_rows, int64_t num_cols,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len, const int src_type, const int etype) {
  // SPMM with CSR.
	printf("SpMMCmpCsrHeteroKernel\n");
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < num_rows) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      using accum_type = typename accum_dtype<DType>::type;
      accum_type local_accum = static_cast<accum_type>(
          out[ty * out_len + tx]);  // ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff =
            BinaryOp::use_lhs ? (ufeat + cid * ufeat_len) : nullptr;
        const DType* eoff =
            BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
        DType tmp_out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(
            &local_accum, &local_argu, &local_arge, tmp_out, cid, eid);
      }
      // Update output only when max/min values are different that original
      // output
      DType new_out = static_cast<DType>(local_accum);
      if (out[ty * out_len + tx] != new_out) {
        out[ty * out_len + tx] = new_out;
        if (ReduceOp::require_arg && BinaryOp::use_lhs) {
          arg_u[ty * out_len + tx] = local_argu;
          arg_u_ntype[ty * out_len + tx] = src_type;
        }
        if (ReduceOp::require_arg && BinaryOp::use_rhs) {
          arg_e[ty * out_len + tx] = local_arge;
          arg_e_etype[ty * out_len + tx] = etype;
        }
      }
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA implementation of g-SpMM on Coo format.
 * @param bcast Broadcast information.
 * @param coo The Coo matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCoo(
    const BcastOff& bcast, const COOMatrix& coo, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  /**
   * TODO(Xin): Disable half precision for SpMMCoo due to the round-off error.
   * We should use fp32 for the accumulation but it's hard to modify the 
   * current implementation.
   */
#if BF16_ENABLED
  if (std::is_same<DType, __half>::value ||
      std::is_same<DType, __nv_bfloat16>::value)
#else
  if (std::is_same<DType, __half>::value)
#endif  // BF16_ENABLED
    LOG(FATAL) << "SpMMCoo doesn't support half precision fow now. "
               << "Please use SpMMCsr instead by allowing the graph "
               << "materialize CSR/CSC formats.";
  const Idx *row = coo.row.Ptr<Idx>(), *col = coo.col.Ptr<Idx>(),
            *edge_map = coo.data.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>(),
              *efeat_data = efeat.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  Idx *argu_data = argu.Ptr<Idx>(), *arge_data = arge.Ptr<Idx>();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const int64_t N = coo.num_rows, M = coo.num_cols, E = coo.row->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;

  int64_t out_size = out.NumElements();
  const int nt = FindNumThreads(out_size);
  const int nb = (out_size + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      _FillKernel, nb, nt, 0, stream, out_data, out_size, ReduceOp::zero());

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(coo.data);

  printf("\n spmmcookernel line 662");
  BCAST_IDX_CTX_SWITCH(bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off, {
    CUDA_KERNEL_CALL(
        (SpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
        arge_data, row, col, edge_map, N, M, E, ubcast_off, ebcast_off, lhs_len,
        rhs_len, len);
    if (ReduceOp::require_arg) {
      CUDA_KERNEL_CALL(
          (ArgSpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
          arge_data, row, col, edge_map, N, M, E, ubcast_off, ebcast_off,
          lhs_len, rhs_len, len);
    }
  });
}

/**
 * @brief CUDA implementation of g-SpMM on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCsr(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const Idx* indptr = csr.indptr.Ptr<Idx>();
  const Idx* indices = csr.indices.Ptr<Idx>();
  const Idx* edge_map = csr.data.Ptr<Idx>();
  const DType* ufeat_data = ufeat.Ptr<DType>();
  const DType* efeat_data = efeat.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  Idx* argu_data = argu.Ptr<Idx>();
  Idx* arge_data = arge.Ptr<Idx>();

  cudaStream_t stream = runtime::getCurrentCUDAStream();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nby = (len + ntx - 1) / ntx;
  const int nbx = FindNumBlocks<'x'>((csr.num_rows + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  printf("\n spmmcsrkernel call line 726");

  BCAST_IDX_CTX_SWITCH(
      bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off,
      {CUDA_KERNEL_CALL(
          (SpMMCsrKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
          arge_data, indptr, indices, edge_map, csr.num_rows, csr.num_cols,
          ubcast_off, ebcast_off, lhs_len, rhs_len, len)});

  cudaEventRecord(stop); // Record stop event after your CUDA operation
  cudaEventSynchronize(stop); // Synchronize on stop event to ensure it has completed

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("\nSpmm kernel time: %.6f seconds\n", milliseconds / 1000);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  }


/**
 * @brief CUDA kernel of SpMM-Min/Max on Csr format on heterogeneous graph.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param argu_ntype Node type of the arg-Min/Max on source nodes, which refers
 * the source node types correspond to the minimum/maximum values of reduction
 * result on destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge_etype Edge-type of the arg-Min/Max on edges. which refers the
 * source node indices correspond to the minimum/maximum values of reduction
 * result on destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param src_type Node type of the source nodes of an etype
 * @param etype Edge type
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCmpCsrHetero(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge, NDArray argu_ntype,
    NDArray arge_etype, const int src_type, const int etype) {
  const Idx* indptr = csr.indptr.Ptr<Idx>();
  const Idx* indices = csr.indices.Ptr<Idx>();
  const Idx* edge_map = csr.data.Ptr<Idx>();
  const DType* ufeat_data = ufeat.Ptr<DType>();
  const DType* efeat_data = efeat.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  Idx* argu_data = argu.Ptr<Idx>();
  Idx* arge_data = arge.Ptr<Idx>();

  cudaStream_t stream = runtime::getCurrentCUDAStream();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((csr.num_rows + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);
  
  printf("spmmcmpcsrhetro kernel line 798");
  BCAST_IDX_CTX_SWITCH(
      bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off,
      {CUDA_KERNEL_CALL(
          (SpMMCmpCsrHeteroKernel<
              Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
          arge_data, static_cast<Idx*>(argu_ntype->data),
          static_cast<Idx*>(arge_etype->data), indptr, indices, edge_map,
          csr.num_rows, csr.num_cols, ubcast_off, ebcast_off, lhs_len, rhs_len,
          len, src_type, etype)});
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_SPMM_CUH_
