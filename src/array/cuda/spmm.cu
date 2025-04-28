/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/spmm.cu
 * @brief SPMM C APIs and definitions.
 */
#include <dgl/array.h>

#include "../../runtime/cuda/cuda_common.h"
#include "./functor.cuh"
#include "./ge_spmm.cuh"
#include "./spmm.cuh"
#include <iostream>
#include <dgl/runtime/device_api.h>  // Required for DLDeviceType
#include <cuda_runtime.h>
namespace dgl {

using namespace cuda;
using namespace dgl::runtime;

namespace aten {
int flag=1;
// CUDA Kernel for element-wise division
__global__ void UpdateEfeatKernel(float* efeat, int64_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Compute global thread index
    if (idx < num_elements) {
        efeat[idx] /= 10000.0f; // Perform division
    }
}

// Host function to launch the CUDA kernel
void UpdateEfeatGPU(float* d_efeat, int64_t num_elements) {
    // Define CUDA thread configuration
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch CUDA kernel
    UpdateEfeatKernel<<<num_blocks, threads_per_block>>>(d_efeat, num_elements);

    // Synchronize to ensure computation completes before returning
    cudaDeviceSynchronize();
}

/**
 * @brief CUDA implementation of g-SpMM on Csr format.
 * @note use cusparse if the reduce operator is `sum` and there is
 *       no broadcast, use dgl's kernel in other cases.
 */
template <int XPU, typename IdType, typename DType>
void SpMMCsr(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  // printf("Inside spmm.cu\n");
  // const auto& size = efeat->shape;
  // int64_t row = size[0];
  // int64_t col = size[1];
  // std::cout<<"row: "<<row<<"col: "<<col<<"\n";
  //  // Print the elements of the 2D NDArray(nodes array)
  // printf("Edge weight\n");
  // float* edge_weight = static_cast<float*>(efeat->data);
  // float host_edge_weight[row][col];
  //  // Copy the data back from GPU to CPU
  // cudaMemcpy(host_edge_weight, edge_weight, row * col * sizeof(float), cudaMemcpyDeviceToHost);
  // for (int64_t i = 0; i < row; ++i) {
  //       for (int64_t j = 0; j < col; ++j) {
  //           std::cout << host_edge_weight[i][j] << ' ';
  //       }
  //       std::cout << std::endl;
  //   }
  // cudaFree(edge_weight);
  // const auto& size1 = ufeat->shape;
  // int64_t row1 = size1[0];
  // int64_t col1 = size1[1];
  // std::cout<<"row: "<<row1<<"col: "<<col1<<"\n";
  //  // Print the elements of the 2D NDArray(nodes array)
  // printf("Node weight\n");
  // float* node_weight = static_cast<float*>(ufeat->data);
  // float host_node_weight[row1][col1];
  //  // Copy the data back from GPU to CPU
  // cudaMemcpy(host_node_weight, node_weight, row1 * col1 * sizeof(float), cudaMemcpyDeviceToHost);
  // for (int64_t i = 0; i < row1; ++i) {
  //       for (int64_t j = 0; j < col1; ++j) {
  //           std::cout << host_node_weight[i][j] << ' ';
  //       }
  //       std::cout << std::endl;
  //   }
  // cudaFree(node_weight);

  //std::cout<<"efeat_size: "<<size<<"\n";
  //printf("efeat size: \n",size);
  // double* edge_weight = static_cast<double*>(efeat->data);
  // double* host_edge_weight = new double[size];  // Allocate space on the host (CPU)
  //
  //    // Copy data from device (GPU) to host (CPU)
  //    cudaMemcpy(host_edge_weight, edge_weight, size * sizeof(double), cudaMemcpyDeviceToHost);
  //
  // printf("Edge weight ");
  // for(int64_t i=0;i<size;i++)
  // {
  //   //printf("%lf ",edge_weight[i]);
  //   std::cout<<host_edge_weight[i]<<"\t";
  // } 
  //printf("\n");
  //delete [] host_edge_weight;
  if(flag == 0){
    //flag++;
  //printing row pointer
  int64_t num_nodes = csr.indptr->shape[0];
  printf("Row pointer size %ld\n",num_nodes);
  const IdType* data = static_cast<IdType*>(GetDevicePointer(csr.indptr));    
     IdType* host_data = new IdType[num_nodes];  // Allocate space on the host (CPU)

     // Copy data from device (GPU) to host (CPU)
     cudaMemcpy(host_data, data, num_nodes * sizeof(IdType), cudaMemcpyDeviceToHost);

     // Now print the data from the host
     // printf("Printing row pointer of CSR\n");
     // for (int64_t i = 0; i < num_nodes; ++i) {
     //     std::cout << host_data[i] <<"\t";
     // }
     // printf("\n");
     // // Clean up
     delete[] host_data;
 
 //printing column index
  int64_t num_nodes1 = csr.indices->shape[0];
  printf("Column indices size %ld\n",num_nodes1);
  const IdType* data1 = static_cast<IdType*>(GetDevicePointer(csr.data));    
     IdType* host_data1 = new IdType[num_nodes1];  // Allocate space on the host (CPU)

     // Copy data from device (GPU) to host (CPU)
     cudaMemcpy(host_data1, data1, num_nodes1 * sizeof(IdType), cudaMemcpyDeviceToHost);

     // Now print the data from the host
     printf("Printing data of CSR\n");
     for (int64_t i = 0; i < num_nodes1; ++i) {
         std::cout << host_data1[i] <<"\t";
     }
      printf("\n");
     // Clean up
     delete[] host_data1;
  }
 
  bool is_scalar_efeat = efeat.NumElements() == csr.indices->shape[0];
  //std::cout<<"is_scalar_efeat: "<<is_scalar_efeat<<"\n";
  bool use_efeat = op != "copy_lhs";
  //std::cout<<"row pointer size: "<<csr.indptr->shape[0]<<" column index size: "<<csr.indices->shape[0]<<" data size: "<<csr.data->shape[0]<<std::endl;
  //std::cout<<is_scalar_efeat;
  // const auto& shape = efeat->shape;
  // int64_t row = shape[0];
  // int64_t col = shape[1];
  // std::cout<<"row"<<row<<"cloumn"<<col;

  if (reduce == "sum") {
    bool more_nnz = (csr.indices->shape[0] > csr.num_rows * csr.num_cols);
    if (op == "copy_lhs" && cusparse_available<DType, IdType>(more_nnz)) {
      // cusparse
      //printf("reduce==sum and op==copy_lhs\n");
      int64_t x_length = 1;
      for (int i = 1; i < ufeat->ndim; ++i) x_length *= ufeat->shape[i];
      // if (!IsNullArray(csr.data)) {
      //   efeat = IndexSelect(efeat, csr.data);
      //   }
      const DType* data2 = static_cast<DType*>(GetDevicePointer(csr.data));
      
      CusparseCsrmm2<DType, IdType>(
          ufeat->ctx, csr, static_cast<DType*>(ufeat->data), nullptr,
          static_cast<DType*>(out->data), x_length);
    } else if (
        op == "mul" && is_scalar_efeat &&
        cusparse_available<DType, IdType>(more_nnz)) {
      // cusparse
      //printf("reduce==sum and op==mul with efeat\n");
      int64_t x_length = 1;
      for (int i = 1; i < ufeat->ndim; ++i) x_length *= ufeat->shape[i];
      if (!IsNullArray(csr.data)) {
        efeat = IndexSelect(efeat, csr.data);
        //printf("Checking spmsr data\n");
      }   
  //efeat = efeat.CopyTo(efeat->ctx) / 10000.0f;
  int64_t num_elements = efeat.NumElements(); // Total number of elements
    float* d_efeat = static_cast<float*>(efeat->data); // Get GPU pointer

    // Call CUDA kernel to modify efeat in-place
    //UpdateEfeatGPU(d_efeat, num_elements);
  /*
  const auto& size = efeat->shape;
  int64_t row = size[0];
  int64_t col = size[1];
  std::cout<<"row: "<<row<<"col: "<<col<<"\n";
   // Print the elements of the 2D NDArray(nodes array)
  printf("Edge weight\n");
  float* edge_weight = static_cast<float*>(efeat->data);
  // int64_t size2 = efeat.NumElements();
  // for (int64_t i = 0; i < size2; ++i) {
  //       edge_weight[i] /= 10000.0f;
  //   }
  float host_edge_weight[row][col];
   // Copy the data back from GPU to CPU
  cudaMemcpy(host_edge_weight, edge_weight, row * col * sizeof(float), cudaMemcpyDeviceToHost);
  for (int64_t i = 0; i < row; ++i) {
        for (int64_t j = 0; j < col; ++j) {
            //std::cout << host_edge_weight[i][j] << ' ';
            //host_edge_weight[i][j] /= 10000;
            printf("%f ",host_edge_weight[i][j]);
        }
        std::cout << std::endl;
    }
   // Copy modified data back to GPU (updates efeat in place)
    //cudaMemcpy(edge_weight, host_edge_weight, row * col * sizeof(float), cudaMemcpyHostToDevice);
  //cudaFree(edge_weight);
  */
    
       CusparseCsrmm2<DType, IdType>(
           ufeat->ctx, csr, static_cast<DType*>(ufeat->data),
          static_cast<DType*>(efeat->data), static_cast<DType*>(out->data),
          x_length);
    } else {  // general kernel
      printf("else\n");
      SWITCH_OP(op, Op, {
        cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Sum<IdType, DType> >(
            bcast, csr, ufeat, efeat, out, NullArray(), NullArray());
      });
    }
  } else if (reduce == "max") {
    printf("reduce==max\n");
    SWITCH_OP(op, Op, {
      cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Max<IdType, DType> >(
          bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else if (reduce == "min") {
    printf("reduce==min\n");
    SWITCH_OP(op, Op, {
      cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Min<IdType, DType> >(
          bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}
/**
 * @brief CUDA implementation of g-SpMM on Coo format.
 */
template <int XPU, typename IdType, typename DType>
void SpMMCoo(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
	//printf("From SpMMCoo\n");
  if (reduce == "sum") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Sum<IdType, DType, true> >(
          bcast, coo, ufeat, efeat, out, NullArray(), NullArray());
    });
  } else if (reduce == "max") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Max<IdType, DType, true> >(
          bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else if (reduce == "min") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Min<IdType, DType, true> >(
          bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

template void SpMMCsr<kDGLCUDA, int32_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCUDA, int64_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
#if BF16_ENABLED
template void SpMMCsr<kDGLCUDA, int32_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCUDA, int64_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
#endif  // BF16_ENABLED
template void SpMMCsr<kDGLCUDA, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCUDA, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCUDA, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCUDA, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);

template void SpMMCoo<kDGLCUDA, int32_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCUDA, int64_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
#if BF16_ENABLED
template void SpMMCoo<kDGLCUDA, int32_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCUDA, int64_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
#endif  // BF16_ENABLED
template void SpMMCoo<kDGLCUDA, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCUDA, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCUDA, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCUDA, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);

}  // namespace aten
}  // namespace dgl
