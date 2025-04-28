/**
 *  Copyright (c) 2021 by Contributors
 * @file array/cuda/rowwise_sampling.cu
 * @brief uniform rowwise sampling
 */

#include <curand_kernel.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/tensordispatch.h>
#include <typeinfo>
#include <numeric>
#include "../../array/cuda/atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "./dgl_cub.cuh"
#include "./utils.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include <dgl/runtime/ndarray.h>
#define NC 20
using namespace dgl::cuda;
using namespace dgl::aten::cuda;
using TensorDispatcher = dgl::runtime::TensorDispatcher;
float sampling_time = 0.0;
namespace dgl {
namespace aten {
namespace impl {

namespace {

//constexpr int BLOCK_SIZE = 64;
constexpr int BLOCK_SIZE = 128;
int64_t* d_part_array;
double* d_seed_features;

//int64_t col;
//int topk_count 
double* d_node_array;
int64_t* nodes_info;
//float sampling_time = 0.0;
int counter = 0;
int64_t n;
//int64_t* d_n;

//int flag = 0;
/**
 * @brief Compute the size of each row in the sampled CSR, without replacement.
 *
 * @tparam IdType The type of node and edge indexes.
 * @param num_picks The number of non-zero entries to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed byconst NDArray& parts_array,
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseSampleDegreeKernel(
  const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  IdType* const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  /* 
  if(tIdx==0)
   {
     printf("Seed nodes ");
     for(int i=0;i<num_rows;i++)
        printf("%lld ",in_rows[i]);
     printf("\n");}
     */
  if (tIdx < num_rows) {
    const int in_row = in_rows[tIdx];
    const int out_row = tIdx;
    out_deg[out_row] = min(
      static_cast<IdType>(num_picks), in_ptr[in_row + 1] - in_ptr[in_row]);

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

/**
 * @brief Compute the size of each row in the sampled CSR, with replacement.
 *
 * @tparam IdType The type of node and edge indexes.const NDArray& parts_array,
 * @param num_picks The number of non-zero entries to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed by
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseSampleDegreeReplaceKernel(
  const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  IdType* const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int64_t in_row = in_rows[tIdx];
    const int64_t out_row = tIdx;

    if (in_ptr[in_row + 1] - in_ptr[in_row] == 0) {
      out_deg[out_row] = 0;
    } else {
      out_deg[out_row] = static_cast<IdType>(num_picks);
    }

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

/**
 * @brief Perform row-wise uniform sampling on a CSR matrix,
 * and generate a COO matrix, without replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_index The indices array of the input CSR.
 * @param data The data array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 */
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel(
  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  const IdType* const in_index, const IdType* const data,
  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
  IdType* const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
    min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        out_idxs[out_row_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(out_idxs + out_row_start + num, idx);
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[perm_idx];
        out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
      }
    }
    out_row += 1;
  }
}



template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel1(
  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  const IdType* const in_index, const IdType* const data,
  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
  IdType* const out_idxs, int64_t* d_part_array) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
    min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  if (blockIdx.x==0 && threadIdx.x == 0)
  {
    //printf("Kishan define function that passed d_part_array and 1st data is :%ld\n",d_part_array[0]);
    //printf("out_row: %ld\n", out_row);
  }
  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        out_idxs[out_row_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(out_idxs + out_row_start + num, idx);
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[perm_idx];
        out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
      }
    }
    out_row += 1;
  }
}

/**
 * @brief Perform row-wise uniform sampling on a CSR matrix,
 * and generate a COO matrix, with replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_index The indices array of the input CSR.
 * @param data The data array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 */
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceKernel(
  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  const IdType* const in_index, const IdType* const data,
  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
  IdType* const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
    min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;

    if (deg > 0) {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const int64_t edge = curand(&rng) % deg;
        const int64_t out_idx = out_row_start + idx;
        out_rows[out_idx] = row;
        out_cols[out_idx] = in_index[in_row_start + edge];
        out_idxs[out_idx] =
          data ? data[in_row_start + edge] : in_row_start + edge;
      }
    }
    out_row += 1;
  }
}

}  // namespace

__device__ double bubble_sort(double* sorted, int n) {
  //printf("Inside dot_product\n");
// Bubble sort in shared memory
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (sorted[j] < sorted[j + 1]) {
                // Swap elements
                int temp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = temp;
            }
        } 
  //return result;
  }
}
__device__ double dot_product(double* vec1,double* vec2, int n) {
  //printf("Inside dot_product\n");
  double result = 0.0;
  for (int i = 0; i < n; ++i) {
    result += vec1[i] * vec2[i];
  }
  return result;
}

// Helper function to calculate magnitude of a vector on GPU
__device__ double magnitude(double* vec, int n) {
  //printf("Inside Magnitude\n");
  double mag = 0.0;
  for (int i = 0; i < n; ++i) {
    mag += vec[i] * vec[i];
  }
  return sqrt(mag);
}

__device__ double similarity(double* row, double* representative, int n)
{
  printf("Inside Similarity\n");
  //Compute dot product
  double dot = dot_product(row,representative,n);

  // Compute magnitudes
  double mag1 = magnitude(row,n);
  double mag2 = magnitude(representative,n);

  // Compute cosine similarity
  if (mag1 == 0 || mag2 == 0) {
    return 0.0; // Handle division by zero
  } else {
    return dot / (mag1 * mag2);
  }
}

//own kernel
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernelSurendra(
  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  const IdType* const in_index, const IdType* const data,
  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
  IdType* const out_idxs, int64_t* cluster_id, double* representative, int64_t* nodes_info, double* seed_features, int64_t n) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);
  //printf("Inside kernel\n");
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  //printf("number of seed nodes %lld inside kernel\n",num_rows);
  // for(int64_t i= tIdx;i<232965;i+=128*1024)
  // printf("%lld\n",nodes_info[i]);
  // if(tIdx == 0)
  // {
  //   printf("Inside kernel\n");
  // }
  /*if(tIdx == 0)
  {
    printf("First element:%d\n",nodes_info[5]);
      printf("start copie nodes\n");
    printf("SSize of array:%lld\n",sizeof(*nodes_info)/sizeof(nodes_info[0]));
    printf("length of array:%lld\n",sizeof(nodes_info));

  } 
  __syncthreads();*/ 
  /* 
  if(tIdx==0)
     {
      for(int k=0;k<20 && num_rows>k;k++){
      p rintf("Neighbors of %lld with cluster id ",in_rows[k]);
      for(int i= in_ptr[in_rows[k]]; i<in_ptr[in_rows[k]+1];i++)
        printf("%lld %lld ",in_index[i],cluster_id[in_index[i]]);
      printf("\n");}
      printf("\n");
     }
  */
  //printf("value of n %lld\n",n);
  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
    min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);
  //printf("out_row=%lld,last_row=%lld\n",out_row,last_row);
  //printf("out_row=%ld,last_row=%ld\n",out_row,last_row);
  int64_t tid = threadIdx.x;
  /*
  if(threadIdx.x==0){
     //printf("%d ",blockIdx.x);
     for(int i=0;i<2708;i++)
     printf("%ld ",cluster_id[i]);}*/
  //if(threadIdx.x==0){
  //printf("num_rows= %ld ",num_rows);}


  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];
    const int64_t in_row_end = in_ptr[row+1]-1;
    //printf("row=%lld,in_row_start=%lld,deg=%lld,out_row_start=%lld,num_picks=%lld\n",row,in_row_start,deg,out_row_start,num_picks);
    //printf("row=%ld,in_row_start=%ld,deg=%ld,out_row_start=%ld,num_picks=%ld\n",row,in_row_start,deg,out_row_start,num_picks);
    //if(threadIdx.x==0)
    //printf("block id %d,node id %ld\n",blockIdx.x,row);


    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        //printf("idx for copy=%d\n",idx);
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
      }
    } else{
      //index_calculation(in_ptr,in_index,in_row_start,deg,cluster_id,index,num_picks);
      // copy permutation over
      __shared__ int cluster[NC];
      __shared__ int len_cluster;
      __shared__ int x1,x2;
      __shared__ int updated_0[NC];
      __shared__ int64_t updater[NC];
      //__shared__ double similarity_array[20];
      //__shared__ double sorted[20];
      //__shared__ double* sorted_ptr;
      extern __shared__ int index[]; //dynamic size array using extern(==fanout)
      __shared__ int cluster_modified[NC];

      if(tid==0)
      {
        len_cluster=0;
        //updater=0;
        //updated_0=0;
        //k=0;
        //m=0;
        //printf("Inside kernel\n");
        //sorted_ptr = sorted;
        //printf("num_picks %ld in_row_start %ld deg %ld \n",num_picks,in_row_start,deg);
        //k=0;
        //printf("cluster id \n");
        //for(int i=0;i<2708;i++)
        //printf("%ld \t",clusters_id[i]);
      }
      if(tid<NC)
      { 
        cluster[tid]=-2;
        updater[tid]=0;
        updated_0[tid]=0;
        //similarity_array[tid]=-2.0;
        //sorted[tid]=-2.0;
      }
      __syncthreads();
      //printf("blockid: %d, threadid: %d, tIdx: %ld\n", blockIdx.x, threadIdx.x, tIdx);
      //printf("cluster id initilized\n");
      for(int tid=threadIdx.x; tid < deg; tid+=BLOCK_SIZE)
      {
        //int length = sizeof(cluster_id) / sizeof(cluster_id[0]);
        int64_t nid = in_index[in_row_start+tid];
        int cluster_id1 = cluster_id[nid];
        //printf("%d\n",cluster_id1);
        if(cluster_id1!=-1)
        {
          if(nodes_info[nid] == 1)
          {
            if(cluster[cluster_id1] == -2)
            {
              cluster[cluster_id1] = nid;
              //printf("By 1\n");
            } 
            else if(updated_0[cluster_id1] == 1)
            {		  
              cluster[cluster_id1] = nid;
              nodes_info[updater[cluster_id1]] = 0;
              updated_0[cluster_id1] = 0;
              updater[cluster_id1] = 0;
              //printf("By 1\n");  
            } 
          }
          else if(cluster[cluster_id1] == -2)
          {
            cluster[cluster_id1] = nid;
            nodes_info[nid] = 1;
            updated_0[cluster_id1] = 1;
            updater[cluster_id1] = nid;
            //printf("By 0\n");
          }
        } 
      } 
      __syncthreads();
      // if(tid<20)
      //       {
      //         printf("block id: %d, thread id: %d, cluster: %d\n",blockIdx.x,threadIdx.x,cluster[tid]);
      //       }
      //printf("Nodeinfo array updated\n");
      if(tid==0)
      {
        for(int i=0;i<NC;i++)
        {
          if(cluster[i]!=-2){
            //atomicAdd(&len_cluster, 1);
            len_cluster++;
            //printf("length calculated\n");
          }
        }
      }
      __syncthreads();
     // if(tid==0)
     //      {
     //        printf("block id: %d, length of cluster: %d\n",blockIdx.x,len_cluster);
     //      }
      //__shared__  double sorted[len_cluster];
      if(len_cluster>0)
      {	   
        /*if(tid==0)
             {
               printf("length of cluster %d\n",len_cluster);
               for(int i=0;i<20;i++)
                    printf("%d \t",cluster[i]);
                    printf("\n");
             }*/
        //__syncthreads();
        //copy 
        /*if(tid<20 && cluster[tid]!=-2)
        {
          int pos = atomicAdd(&k, 1);
          cluster_modified[pos] = cluster[tid];

        }*/
        //printf("modified cluster created\n");

        if(tid==0)
          {
            int k=0;
            for(int i=0;i<NC;i++)
            {
              if(cluster[i]!=-2)
                cluster_modified[k++] = cluster[i];
            }
          }
        //printf("length calculated\n");
        /* if(tid==0)
             {
               printf("length of cluster %d\n",len_cluster);
               for(int i=0;i<len_cluster;i++)
                    printf("%d \t",cluster_modified[i]);
                    printf("\n");
             }*/
        //printf("cluster array created\n");
        if(tid == 0)
        {
          x1 = num_picks/len_cluster;
          x2 = num_picks%len_cluster;
        }
        __syncthreads();
        //printf("inside the cluster making\n");
        //if(tid < 20)
        //{
          if(len_cluster < num_picks)
          {
            if(tid < x1)
            {
              for(int j=0; j<len_cluster; j++)
              {
                index[j+tid*len_cluster]=cluster_modified[j];
              }
            }
            //__syncthreads();
            if(tid<x2)
            {
              index[len_cluster*x1+tid] = cluster_modified[tid];
              //printf("data coppied in index\n");
            }
          }
          else if(len_cluster == num_picks)
          {
            if(tid<len_cluster)
            {
              index[tid] = cluster_modified[tid];
            }
          }
          else
        {
            // if(tid<num_picks)
            // {
            // index[tid]=cluster_modified[tid];
            // } 
           __shared__ double similarity_array[NC];
           __shared__ double sorted[NC];
           __shared__ float epsilon; 
          if(tid == 0)
          {
            epsilon = 0.000001;
          }

            //printf("Inside else part\n");
             if(tid<20)
              { 
                //cluster[tid]=-2;
                //similarity_array[tid]=static_cast<double>(tid);
                similarity_array[tid] = -2.0;
                sorted[tid]=-2.0;
              }
      __syncthreads();


            if(tid<NC)
          {
              if(cluster[tid]!=-2)
            {
              //printf("calculating similarit\n");
              //similarity_array[tid] = similarity(&seed_features[n*out_row],&representative[n*tid], n);
              double dot = dot_product(&seed_features[n*out_row],&representative[n*tid],n);


              // Compute magnitudes
              double mag1 = magnitude(&seed_features[n*out_row],n);
              double mag2 = magnitude(&representative[n*tid],n);

              // Compute cosine similarity
              if (mag1 == 0.0 || mag2 == 0.0) {
                similarity_array[tid] = 0.0; // Handle division by zero
              } else {
                similarity_array[tid] = (dot / (mag1 * mag2));
              }
              //printf("one thread completed calculating similarity\n");
            }
          }
            __syncthreads();
          // if(tid<20)
          // {
          //   printf("block id: %d, thread id: %d, similarity: %f\n",blockIdx.x,threadIdx.x,similarity_array[tid]);
          // }
          if(tid==0)
          {
            int p=0;
            for(int s=0;s<NC;s++)
            {
              if(similarity_array[s]!=-2.0)
                sorted[p++] = similarity_array[s];
            }
          }
          __syncthreads();

           // if(tid<20)
           // {
           //   if(sorted[tid]==0)
           //   printf("block id: %d, thread id: %d, similarity: %f\n",blockIdx.x,threadIdx.x,sorted[tid]);
           // }
           if(tid==0)
          {
          for (int i = 0; i < len_cluster-1; i++) {
           for (int j = i+1; j < len_cluster; j++) {
            if (sorted[i] < sorted[j]) {
                // Swap elements
                  //printf("block id: %d, thread id: %d, similarity[i]: %f, similarity[j]: %f\n",blockIdx.x,threadIdx.x,sorted[i], sorted[j]);
                float temp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = temp;
                //printf("bid: %d, i: %f, j: %f, temp: %f \n", blockIdx.x, sorted[i], sorted[j], temp);
                }
               } 
              //return result;
              }

            //bubble_sort(sorted,20);
          }
        __syncthreads();
           // if(tid<20)
           // {
           //   printf("block id: %d, thread id: %d, similarity: %f\n",blockIdx.x,threadIdx.x,sorted[tid]);
           // }
          if(tid < num_picks)
            {
              //float epsilon = 0.000001;
              float x = sorted[tid];
              for(int r=0; r<NC; r++)
                if(fabs(similarity_array[r] - x) < epsilon)
                  {
                  index[tid] = cluster[r];
                  //break;
                  //printf("x=%f, neighbor choosen: %d\n",x,cluster[r]);
                  }
              //printf("index updated\n");
            }
          //
        }
        //}
      }
      //length_cluster is zero
      else {
        if(tid < num_picks)
        {
          index[tid] = in_index[in_row_start+tid];
        }
      }		   

      //	__syncthreads();
      //if(threadIdx.x==0){
      //printf("index of %d\t",blockIdx.x);
      //for(int i=0;i<num_picks;i++)
      //if(threadIdx.x<num_picks)
      // printf("index of: %d Tid: %d is %d \n",blockIdx.x, threadIdx.x, index[threadIdx.x]);
      //  }
      /* if(threadIdx.x==0){
        for(int i=0;i<num_rows;i++)
           if(blockIdx.x==i)
          {
        //printf("block id %d \t",blockIdx.x);
          for(int k=0;k<num_picks;k++)
        printf("block id %d: %d \t",blockIdx.x,index[k]);
        }
           else
          for (long long int j = 0; j < 10000000000; ++j) {
                              // Idle loop
                                  }
        }*/
      //printf("start copie nodes\n");

      __syncthreads();
      //printf("start copie nodes\n");

      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        //printf("idx for computation=%d\n",idx);
        //const IdType in_idx = in_row_start + idx;
        //printf("perm_idx=%ld\n",perm_idx);
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = index[idx];
        out_idxs[out_row_start + idx] = data ? data[idx] : idx;
      }
    }
    out_row += 1;
  }
}


///////////////////////////// CSR sampling //////////////////////////

template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const int64_t num_rows = rows->shape[0];
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
    ? static_cast<IdType*>(GetDevicePointer(mat.data))
    : nullptr;


  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);
  // 
  // printf("data from rowwise_sampling.cu line 265\n")
  // for(int i=0; i< size; i++)
  // {
  //     std::cout << part_array[i] << " ";
  // }
  // // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(cuda_array, data_ptr, size * sizeof(int64_t), cudaMemcpyHostToDevice);


  // compute degree
  IdType* out_deg = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  }

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  }

  // copy using the internal current stream
  CUDA_CALL(cudaMemcpyAsync(
    new_len_tensor->data, out_ptr + num_rows, sizeof(IdType),
    cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  //			cudaEvent_t start,stop;
  //				cudaEventCreate(&start);
  //				cudaEventCreate(&stop);
  constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  //cudaEventRecord(start);
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    // cudaEventRecord(start);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
      0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
      in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
    // cudaEventRecord(stop);
  } else {  // without replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, 0,
      stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
      data, out_ptr, out_rows, out_cols, out_idxs);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    sampling_time += milliseconds/1000;
    printf("cuda sampling time %.6f\n", sampling_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  //				cudaDeviceSynchronize();
  // cudaEventRecord(stop);
  //				cudaEventSynchronize(stop);
  device->FreeWorkspace(ctx, out_ptr);
  // float milliseconds = 0;
  //	cudaEventElapsedTime(&milliseconds, start, stop);
  // printf("cuda sapmling time: %.6f\n", milliseconds/1000);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  const IdType new_len = static_cast<const IdType*>(new_len_tensor->data)[0];
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  return COOMatrix(
    mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform1(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, NDArray part_array, const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  size_t size = part_array->shape[0];
  //size_t size1 = node_array->shape[0];
  cudaMalloc((void **)&nodes_info, size * sizeof(int64_t));
  cudaMemset(nodes_info, 0, size * sizeof(int64_t));
  cudaDeviceSynchronize();
  /*
  printf("size of part_array %zu \n",size);
  printf("Size of nodes info array %lld \n",sizeof(*nodes_info)/sizeof(nodes_info[0]));
  printf("Size of nodes info %lld \n",sizeof(nodes_info));
  */
  //int64_t size = part_array->shape[0];
  //cudaEvent_t start, stop;
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);

  //cudaEventRecord(start);
  //printf("From _CSRRowWiseSamplingUniform1\n");
  int64_t* parts_array = static_cast<int64_t*>(part_array->data);
  //int64_t* nodes_array = static_cast<int64_t*>(node_array->data);
  /* if(flag == 0){
  for(int64_t i=0; i< size; i++)
   {
       std::cout << parts_array[i] << "\n";
   }
} */
  //cudaEventRecord(stop); // Record stop event after your CUDA operation
  //cudaEventSynchronize(stop);

  //float milliseconds = 0;
  //cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("\npart_array->data to parts_array copy time: %.6f seconds\n", milliseconds / 1000);

  //cudaEventDestroy(start);
  //cudaEventDestroy(stop);

  //printf("size %zu \n",size);

  // printf("data from _CSRRowWiseSamplingUniform1 function line 448");
  //int<t_k\80>ýrequire"cmp.utils.feedkeys".run(7)
  //u64_t* d_part_array = parts_array;
  //printf("Last element of part_array is %ld \n",parts_array[size-1]);
  // int64_t* d_part_array;
  cudaMalloc(&d_part_array, size * sizeof(int64_t));
  cudaMemcpy(d_part_array, parts_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);
  //cudaMalloc(&d_node_array, size1 * sizeof(int64_t));
  //cudaMemcpy(d_node_array, nodes_array, size1 * sizeof(int64_t), cudaMemcpyHostToDevice);

  //printf("cudaMemcpy called\n");


  /*
  int64_t* d_part_array;
  if(!(parts_array[size-1])){
  //int64_t* d_part_array;

  // allocate gpu memory
  // IdType* d_part_array = static_cast<IdType*>(device->AllocWorkspace(ctx, (size) * sizeof(IdType)));
  //cudaEvent_t start, stop;
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);
//
  cudaEventRecord(start);
  cudaMalloc(&d_part_array, size * sizeof(int64_t));
  cudaMemcpy(d_part_array, parts_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);
  }*/
  //cudaEventRecord(stop); // Record stop event after your CUDA operation
  //cudaEventSynchronize(stop); // Synchronize on stop event to ensure it has completed

  //float milliseconds = 0;
  //cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("\ncpu to gpu copy time(cluster id): %.6f seconds\n", milliseconds / 1000);

  //cudaEventDestroy(start);
  //cudaEventDestroy(stop);

  const int64_t num_rows = rows->shape[0];
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);
  //printf("num_rows=%ld,num_picks=%ld\n",num_rows,num_picks);

  IdArray picked_row =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
    ? static_cast<IdType*>(GetDevicePointer(mat.data))
    : nullptr;
  const int64_t num_nodes = mat.num_rows;
  //printf("Number of elements in graph %lld\n",num_nodes);
  //for(long long int i=0; i<num_nodes; i++)
  //printf("%lld \n",parts_array[i]);  


  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);
  // 
  // printf("data from rowwise_sampling.cu line 478\n");
  // for(int i=0; i< size; i++)
  // {
  //     std::cout << part_array[i] << " ";
  // }
  // // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(cuda_array, data_ptr, size * sizeof(int64_t), cudaMemcpyHostToDevice);


  // compute degree
  IdType* out_deg = static_cast<IdType*>(device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  //IdType* index = static_cast<IdType*>(device->AllocWorkspace(ctx, 10* sizeof(IdType)));
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);

  //cudaEventRecord(start);
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  }
  //cudaEventRecord(stop); // Record stop event after your CUDA operation
  //cudaEventSynchronize(stop); // Synchronize on stop event to ensure it has completed

  //milliseconds = 0;
  //cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("\ndegree kernel time: %.6f seconds\n", milliseconds / 1000);

  //cudaEventDestroy(start);
  //cudaEventDestroy(stop);

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  }

  // copy using the internal current stream
  CUDA_CALL(cudaMemcpyAsync(
    new_len_tensor->data, out_ptr + num_rows, sizeof(IdType),
    cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  //	cudaEvent_t start,stop;
  //		cudaEventCreate(&start);
  //		cudaEventCreate(&stop);
  //constexpr int TILE_SIZE = 64 / BLOCK_SIZE;
  constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    //cudaEventRecord(start);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
      0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
      in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
    // cudaEventRecord(stop);
  } else {  // without replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    /*
     CUDA_KERNEL_CALL(
         (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, 0,
         stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
         data, out_ptr, out_rows, out_cols, out_idxs);*/

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    /*CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformKernelSurendra<IdType, TILE_SIZE>), grid, block, num_picks,
      stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
      data, out_ptr, out_rows, out_cols, out_idxs, d_part_array, d_node_array, nodes_info, col);
    */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    sampling_time += milliseconds/1000;
    //printf("cuda sampling time %.6f\n", sampling_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


  }

  // 3rd parameter is shared memory size
  //cudaDeviceSynchronize();
  //cudaEventRecord(stop);
  //cudaEventSynchronize(stop);
  device->FreeWorkspace(ctx, out_ptr);
  //device->FreeWorkspace(ctx, index);
  //float milliseconds = 0;
  //cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("cuda sapmling time: %.6f\n", milliseconds/1000);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  //cudaFree(d_part_array);
  const IdType new_len = static_cast<const IdType*>(new_len_tensor->data)[0];
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);
  //printf("new length %lld\n",new_len);
  //printf("num_rows %lld, num_cols %lld\n",mat.num_rows,mat.num_cols);
  /*
  const int64_t picked_row_size = picked_row->shape[0];
  printf("Picked_row_size:%lld\n",picked_row_size);
  const int64_t picked_col_size = picked_col->shape[0];
  printf("Picked_col_size:%lld\n",picked_col_size);
  */
  /*
  IdType* column;
   IdType* row;
  column=(IdType*)malloc(new_len * sizeof(IdType));
  row=(IdType*)malloc(new_len * sizeof(IdType));
  cudaMemcpy(row,out_rows, new_len * sizeof(IdType), cudaMemcpyDeviceToHost);
  cudaMemcpy(column,out_cols, new_len * sizeof(IdType), cudaMemcpyDeviceToHost);
  printf("out_rows ");
  for(int j=0;j<new_len;j++)
    printf("%lld ",row[j]);
    printf("\n");
  printf("out_columns ");
  for(int j=0;j<new_len;j++)
    printf("%lld ",column[j]);
    printf("\n");
  */   

  // cudaFree(d_part_array);
  return COOMatrix(
    mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}


template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform2(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, NDArray seed_features, const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  //printf("From _CSRRowWiseSamplingUniform2\n");
  // size_t size = part_array->shape[0];
  // int64_t* parts_array = static_cast<int64_t*>(part_array->data);

  // printf("data from _CSRRowWiseSamplingUniform2 function line 448");
  // if(flag == 0)
  // {
  // printf("flag activated");
  // flag = 1;
  // allocate gpu memory
  // IdType* d_part_array = static_cast<IdType*>(device->AllocWorkspace(ctx, (size) * sizeof(IdType)));
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(d_part_array, parts_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);
  // printf("Cudamemcpy called");
  // flag = 1;
  // }
  printf("Layer two started\n");
  const auto& shape1 = seed_features->shape;
  int64_t row1 = shape1[0];
  int64_t col1 = shape1[1];
  printf("rows and cols  of seed_features are %lld %lld in second\n",row1,col1);
  double* seed_features1 = static_cast<double*>(seed_features->data);
  // Print the elements of the 2D NDArray(seed features)
  /*printf("seed node feature in second\n");
  for(int64_t i = 0; i < row1; ++i) {
    for(int64_t j = 0; j < col1; ++j) {
      std::cout << seed_features1[i * col1 + j] << ' ';
    }
    std::cout << std::endl;
  }*/
  printf("Before saving seed features in GPU\n");
  double* d_seed_features;
  cudaError_t err = cudaMallocManaged((void**)&d_seed_features, row1 * col1 * sizeof(double));
  if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "cudaMalloc successful!" << std::endl;
    }
  printf("memory is allocated for seed features\n");
  //double* d_seed_features = static_cast<double*>(device->AllocWorkspace(ctx,  row1 * col1 * sizeof(double)));
  err = cudaMemcpy(d_seed_features, seed_features1, row1 * col1 * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "cudaMemcpy successful!" << std::endl;
    }
  printf("After saving seed features in GPU\n");

  const int64_t num_rows = rows->shape[0];
  printf("Number of seed nodes %lld in second\n",num_rows);
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
    ? static_cast<IdType*>(GetDevicePointer(mat.data))
    : nullptr;
  //printf("Inside uniform2\n");
  counter++;
  if(counter==3)
  {
    const int64_t size = mat.num_rows;
    //printf("Counter: %d \t Graph size: %lld \n",counter,size);
    //cudaMalloc((void **)&nodes_info, size * sizeof(int64_t));
    cudaMemset(nodes_info, 0, size * sizeof(int64_t));
    cudaDeviceSynchronize();
    counter=0;
  }  



  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);
  // 
  // printf("data from rowwise_sampling.cu line 478\n");
  // for(int i=0; i< size; i++)
  // {
  //     std::cout << part_array[i] << " ";
  // }
  // // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(cuda_array, data_ptr, size * sizeof(int64_t), cudaMemcpyHostToDevice);


  // compute degree
  IdType* out_deg = static_cast<IdType*>(device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  }

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  }

  // copy using the internal current stream
  CUDA_CALL(cudaMemcpyAsync(
    new_len_tensor->data, out_ptr + num_rows, sizeof(IdType),
    cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  //constexpr int TILE_SIZE = 64 / BLOCK_SIZE;
  constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    //cudaEventRecord(start);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
      0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
      in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
    // cudaEventRecord(stop);
  } else {  // without replacement
    // const dim3 block(BLOCK_SIZE);
    // const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    // CUDA_KERNEL_CALL(
    //     (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, 0,
    //     stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
    //     data, out_ptr, out_rows, out_cols, out_idxs);
    //
    //printf("calling kernel\n");
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /*CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformKernelSurendra<IdType, TILE_SIZE>), grid, block, num_picks,
      stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
      data, out_ptr, out_rows, out_cols, out_idxs, d_part_array, d_node_array, nodes_info, d_seed_features, col1);*/
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    sampling_time += milliseconds/1000;
    //printf("cuda sampling time %.6f\n", sampling_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

  }

  device->FreeWorkspace(ctx, out_ptr);
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  // cudaDeviceSynchronize();
  // wait for copying `new_len` to finish
  // cudaFree(d_part_array);
  cudaFree(d_seed_features);
  const IdType new_len = static_cast<const IdType*>(new_len_tensor->data)[0];
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);
  //printf("new length %lld\n",new_len);
  //printf("num_rows %lld, num_cols %lld\n",mat.num_rows,mat.num_cols);
  /*
  const int64_t picked_row_size = picked_row->shape[0];
  printf("Picked_row_size:%lld\n",picked_row_size);
  const int64_t picked_col_size = picked_col->shape[0];
  printf("Picked_col_size:%lld\n",picked_col_size);
  */
  return COOMatrix(
    mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform3(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, NDArray part_array, NDArray node_array, NDArray seed_features, const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  size_t size = part_array->shape[0];
  size_t size1 = node_array->shape[0];
  const auto& shape = node_array->shape;
  int64_t row = shape[0];
  int64_t col = shape[1];
  //printf("First layer starts");
  //printf("rows and cols of representative are %lld %lld\n",row,col);
  const auto& shape1 = seed_features->shape;
  int64_t row1 = shape1[0];
  int64_t col1 = shape1[1];
  //printf("rows and cols of seed_features are %lld %lld\n",row1,col1);
  n=col1;
  //printf("Inside  _CSRRowWiseSamplingUniform3\n");

  cudaMalloc((void **)&nodes_info, size * sizeof(int64_t));
  cudaMemset(nodes_info, 0, size * sizeof(int64_t));
  //cudaDeviceSynchronize();
  /*
  printf("size of part_array %zu \n",size);
  printf("Size of nodes info array %lld \n",sizeof(*nodes_info)/sizeof(nodes_info[0]));
  printf("Size of nodes info %lld \n",sizeof(nodes_info));
  */
  //int64_t size = part_array->shape[0];
  //cudaEvent_t start, stop;
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);

  //cudaEventRecord(start);
  //printf("From _CSRRowWiseSamplingUniform1\n");
  int64_t* parts_array = static_cast<int64_t*>(part_array->data);
  double* nodes_array = static_cast<double*>(node_array->data);
  double* seed_features1 = static_cast<double*>(seed_features->data);
  //std::cout << "Type of node_array: " << typeid(nodes_array).name() << std::endl;
  //printf("nodes_array data \n");
  /*
  for (int64_t i = 0; i < row; ++i) {
        for (int64_t j = 0; j < col; ++j) {
            nodes_array[i * col + j] = static_cast<double>(i * col + j + 1);
        }
    }
    */
  // Print the elements of the 2D NDArray(seed features)
  /*printf("seed node feature\n");
  for (int64_t i = 0; i < row1; ++i) {
    for (int64_t j = 0; j < col1; ++j) {
      std::cout << seed_features1[i * col + j] << ' ';
    }
    std::cout << std::endl;
  }*/
  // Print the elements of the 2D NDArray(nodes array)
  /*for (int64_t i = 0; i < row; ++i) {
        for (int64_t j = 0; j < col; ++j) {
            std::cout << nodes_array[i * col + j] << ' ';
        }
        std::cout << std::endl;
    }*/

  /* if(flag == 0){
  for(int64_t i=0; i< size; i++)
   {
       std::cout << parts_array[i] << "\n";
   }
} */
  /*for(int64_t i=0; i< size1; i++)
    {
       std::cout << nodes_array[i] << "\n";
    }*/
  //cudaEventRecord(stop); // Record stop event after your CUDA operation
  //cudaEventSynchronize(stop);

  //float milliseconds = 0;
  //cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("\npart_array->data to parts_array copy time: %.6f seconds\n", milliseconds / 1000);

  //cudaEventDestroy(start);
  //cudaEventDestroy(stop);

  //printf("size %zu \n",size);

  // printf("data from _CSRRowWiseSamplingUniform1 function line 448");
  //int<t_k\80>ýrequire"cmp.utils.feedkeys".run(7)
  //u64_t* d_part_array = parts_array;
  //printf("Last element of part_array is %ld \n",parts_array[size-1]);
  // int64_t* d_part_array;
  //double* d_seed_features;
  cudaMalloc(&d_part_array, size * sizeof(int64_t));
  cudaMemcpy(d_part_array, parts_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMalloc(&d_node_array, row * col * sizeof(double));
  cudaMemcpy(d_node_array, nodes_array, row * col * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err = cudaMalloc(&d_seed_features, row1 * col1 * sizeof(double));
  if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        //std::cout << "cudaMalloc successful!" << std::endl;
    }
  //double* d_seed_features = static_cast<double*>(device->AllocWorkspace(ctx, (row1*col1) * sizeof(double)));
  err = cudaMemcpy(d_seed_features, seed_features1, row1 * col1 * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        //std::cout << "cudaMemcpy successful!" << std::endl;
    }

  //printf("cudaMemcpy called\n");


  /*
  int64_t* d_part_array;
  if(!(parts_array[size-1])){
  //int64_t* d_part_array;

  // allocate gpu memory
  // IdType* d_part_array = static_cast<IdType*>(device->AllocWorkspace(ctx, (size) * sizeof(IdType)));
  //cudaEvent_t start, stop;
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);
//
  cudaEventRecord(start);
  cudaMalloc(&d_part_array, size * sizeof(int64_t));
  cudaMemcpy(d_part_array, parts_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);
  }*/
  //cudaEventRecord(stop); // Record stop event after your CUDA operation
  //cudaEventSynchronize(stop); // Synchronize on stop event to ensure it has completed

  //float milliseconds = 0;
  //cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("\ncpu to gpu copy time(cluster id): %.6f seconds\n", milliseconds / 1000);

  //cudaEventDestroy(start);
  //cudaEventDestroy(stop);

  const int64_t num_rows = rows->shape[0];
  //printf("Number of seed nodes %lld\n",num_rows);

  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);
  //const int64_t* const slice_rows1 = static_cast<const int64_t*>(rows->data);
  //printf("num_rows=%ld,num_picks=%ld\n",num_rows,num_picks);
  //for(int64_t i=0;i<num_rows;i++)
    //printf("%lld ",slice_rows1);
  //printf("\n");

  IdArray picked_row =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
    ? static_cast<IdType*>(GetDevicePointer(mat.data))
    : nullptr;
  const int64_t num_nodes = mat.num_rows;
  //printf("Number of elements in graph %lld\n",num_nodes);
  //for(long long int i=0; i<num_nodes; i++)
  //printf("%lld \n",parts_array[i]);  


  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);
  // 
  // printf("data from rowwise_sampling.cu line 478\n");
  // for(int i=0; i< size; i++)
  // {
  //     std::cout << part_array[i] << " ";
  // }
  // // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(cuda_array, data_ptr, size * sizeof(int64_t), cudaMemcpyHostToDevice);


  // compute degree
  IdType* out_deg = static_cast<IdType*>(device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  //IdType* index = static_cast<IdType*>(device->AllocWorkspace(ctx, 10* sizeof(IdType)));
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);

  //cudaEventRecord(start);
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  }
  //cudaEventRecord(stop); // Record stop event after your CUDA operation
  //cudaEventSynchronize(stop); // Synchronize on stop event to ensure it has completed

  //milliseconds = 0;
  //cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("\ndegree kernel time: %.6f seconds\n", milliseconds / 1000);

  //cudaEventDestroy(start);
  //cudaEventDestroy(stop);

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  }

  // copy using the internal current stream
  CUDA_CALL(cudaMemcpyAsync(
    new_len_tensor->data, out_ptr + num_rows, sizeof(IdType),
    cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  //	cudaEvent_t start,stop;
  //		cudaEventCreate(&start);
  //		cudaEventCreate(&stop);
  //constexpr int TILE_SIZE = 64 / BLOCK_SIZE;
  constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    //cudaEventRecord(start);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
      0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
      in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
    // cudaEventRecord(stop);
  } else {  // without replacement
    //printf("calling the kernel\n");
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    /*
     CUDA_KERNEL_CALL(
       (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, 0,
         stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
         data, out_ptr, out_rows, out_cols, out_idxs);*/

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformKernelSurendra<IdType, TILE_SIZE>), grid, block, num_picks,
      stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
      data, out_ptr, out_rows, out_cols, out_idxs, d_part_array, d_node_array, nodes_info, d_seed_features,n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    sampling_time += milliseconds/1000;
    //milliseconds = milliseconds/1000;
    //printf("cuda sampling time %.6f\n",milliseconds/1000);

    printf("cuda sampling time %.6f\n", sampling_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


  }

  // 3rd parameter is shared memory size
  //cudaDeviceSynchronize();
  //cudaEventRecord(stop);
  //cudaEventSynchronize(stop);
  device->FreeWorkspace(ctx, out_ptr);
  //device->FreeWorkspace(ctx, index);
  //float milliseconds = 0;
  //cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("cuda sapmling time: %.6f\n", milliseconds/1000);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  //cudaFree(d_part_array);
  //cudaFree(d_seed_features);
  const IdType new_len = static_cast<const IdType*>(new_len_tensor->data)[0];
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);
  //printf("new length %lld\n",new_len);
  //printf("num_rows %lld, num_cols %lld\n",mat.num_rows,mat.num_cols);
  /*
  const int64_t picked_row_size = picked_row->shape[0];
  printf("Picked_row_size:%lld\n",picked_row_size);
  const int64_t picked_col_size = picked_col->shape[0];
  printf("Picked_col_size:%lld\n",picked_col_size);
  */
  /*
  IdType* column;
   IdType* row;
  column=(IdType*)malloc(new_len * sizeof(IdType));
  row=(IdType*)malloc(new_len * sizeof(IdType));
  cudaMemcpy(row,out_rows, new_len * sizeof(IdType), cudaMemcpyDeviceToHost);
  cudaMemcpy(column,out_cols, new_len * sizeof(IdType), cudaMemcpyDeviceToHost);
  printf("out_rows ");
  for(int j=0;j<new_len;j++)
    printf("%lld ",row[j]);
    printf("\n");
  printf("out_columns ");
  for(int j=0;j<new_len;j++)
    printf("%lld ",column[j]);
    printf("\n");
  */   

  // cudaFree(d_part_array);
  
        // Handle int64_t
  //std::cout<<picked_col->shape[0]<<std::endl;
  //std::cout<<"number of row and column for coo "<<mat.num_rows<<"\t"<<mat.num_cols<<std::endl;

        // int64_t* data_ptr = static_cast<int64_t*>(picked_col->data);
         //  for(int64_t i = 0; i < picked_col->shape[0]; ++i) {
         //      std::cout << picked_col->data[i] << " ";
         // }
    //std::cout << std::endl;

  return COOMatrix(
    mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform4(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  //printf("Inside _CSRRowWiseSamplingUniform4\n");
  //printf("From _CSRRowWiseSamplingUniform2\n");
  // size_t size = part_array->shape[0];
  // int64_t* parts_array = static_cast<int64_t*>(part_array->data);

  // printf("data from _CSRRowWiseSamplingUniform2 function line 448");
  // if(flag == 0)
  // {
  // printf("flag activated");
  // flag = 1;
  // allocate gpu memory
  // IdType* d_part_array = static_cast<IdType*>(device->AllocWorkspace(ctx, (size) * sizeof(IdType)));
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(d_part_array, parts_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);
  // printf("Cudamemcpy called");
  // flag = 1;
  // }
  const int64_t num_rows = rows->shape[0];
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
    ? static_cast<IdType*>(GetDevicePointer(mat.data))
    : nullptr;
  //printf("Inside uniform2\n");
  counter++;
  if(counter==3)
  {
   const int64_t size = mat.num_rows;
   //printf("Counter: %d \t Graph size: %lld \n",counter,size);
   //cudaMalloc((void **)&nodes_info, size * sizeof(int64_t));
   cudaMemset(nodes_info, 0, size * sizeof(int64_t));
   cudaDeviceSynchronize();
   counter=0;
  }



  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);
  //
  // printf("data from rowwise_sampling.cu line 478\n");
  // for(int i=0; i< size; i++)
  // {
  //     std::cout << part_array[i] << " ";
  // }
  // // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(cuda_array, data_ptr, size * sizeof(int64_t), cudaMemcpyHostToDevice);


  // compute degree
  IdType* out_deg = static_cast<IdType*>(device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  }

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  }

  // copy using the internal current stream
  CUDA_CALL(cudaMemcpyAsync(
    new_len_tensor->data, out_ptr + num_rows, sizeof(IdType),
    cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  //constexpr int TILE_SIZE = 64 / BLOCK_SIZE;
  constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    //cudaEventRecord(start);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
      0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
      in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
    // cudaEventRecord(stop);
  } else {  // without replacement
    // const dim3 block(BLOCK_SIZE);
    // const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    // CUDA_KERNEL_CALL(
    //     (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, 0,
    //     stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
    //     data, out_ptr, out_rows, out_cols, out_idxs);
    //
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformKernelSurendra<IdType, TILE_SIZE>), grid, block, num_picks,
      stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
      data, out_ptr, out_rows, out_cols, out_idxs, d_part_array, d_node_array, nodes_info, d_seed_features, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    sampling_time += milliseconds/1000;
    printf("cuda sampling time %.6f\n", sampling_time);
    //printf("cuda sampling time %.6f\n",milliseconds/1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

  }

  device->FreeWorkspace(ctx, out_ptr);
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  // cudaDeviceSynchronize();
  // wait for copying `new_len` to finish
  // cudaFree(d_part_array);

  const IdType new_len = static_cast<const IdType*>(new_len_tensor->data)[0];
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);
  //printf("new length %lld\n",new_len);
  //printf("num_rows %lld, num_cols %lld\n",mat.num_rows,mat.num_cols);
  /*
  const int64_t picked_row_size = picked_row->shape[0];
  printf("Picked_row_size:%lld\n",picked_row_size);
  const int64_t picked_col_size = picked_col->shape[0];
  printf("Picked_col_size:%lld\n",picked_col_size);
  */
  return COOMatrix(
    mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {

  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);

  // printf("data from rowwise_sampling.cu line 265\n");
  // for(int i=0; i< size; i++)
  // {
  // std::cout << part_array[i] << " ";
  // }

  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
      mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    return _CSRRowWiseSamplingUniform<XPU, IdType>(
      mat, rows, num_picks, replace);
  }
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform1(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const NDArray& parts_array, const bool replace) {


  // size_t size = parts_array->shape[0];
  // IdArray* part_array = static_cast<IdArray*>(parts_array->data);
  //
  // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));
  //
  // cudaMemcpy(d_part_array, part_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);

  size_t size = parts_array->shape[0];
  int64_t* part_array = static_cast<int64_t*>(parts_array->data);

  // printf("data from rowwise_sampling.cu line 265\n");
  // for(int i=0; i< size; i++)
  // {
  // std::cout << "data :" << part_array[i] << " ";
  // }
  // dgl::IdArray part_array = dgl::aten::AsIdArray(parts_array);
  // dgl::IdArray part_array = dgl::IdArray::FromDLPack(parts_array.ToDLPack());

  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
      mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    return _CSRRowWiseSamplingUniform1<XPU, IdType>(
      mat, rows, num_picks, parts_array, replace);
  }
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform2(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const NDArray& seed_features, const bool replace) {
  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
      mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    return _CSRRowWiseSamplingUniform2<XPU, IdType>(
      mat, rows, num_picks, seed_features, replace);
  }
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform4(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {
  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
      mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    return _CSRRowWiseSamplingUniform4<XPU, IdType>(
      mat, rows, num_picks, replace);
  }
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform3(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const NDArray& parts_array, const NDArray& nodes_array, const NDArray& seed_features, const bool replace) {


  // size_t size = parts_array->shape[0];
  // IdArray* part_array = static_cast<IdArray*>(parts_array->data);
  //
  // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));
  //
  // cudaMemcpy(d_part_array, part_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);

  size_t size = parts_array->shape[0];
  int64_t* part_array = static_cast<int64_t*>(parts_array->data);


  // printf("data from rowwise_sampling.cu line 265\n");
  // for(int i=0; i< size; i++)
  // {
  // std::cout << "data :" << part_array[i] << " ";
  // }
  // dgl::IdArray part_array = dgl::aten::AsIdArray(parts_array);
  // dgl::IdArray part_array = dgl::IdArray::FromDLPack(parts_array.ToDLPack());

  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
      mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    return _CSRRowWiseSamplingUniform3<XPU, IdType>(
      mat, rows, num_picks, parts_array, nodes_array, seed_features, replace);
  }
}

template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int32_t>(
  CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int64_t>(
  CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform1<kDGLCUDA, int32_t>(
  CSRMatrix, IdArray, int64_t, const NDArray&, bool);
template COOMatrix CSRRowWiseSamplingUniform1<kDGLCUDA, int64_t>(
  CSRMatrix, IdArray, int64_t, const NDArray&, bool);
template COOMatrix CSRRowWiseSamplingUniform2<kDGLCUDA, int32_t>(
  CSRMatrix, IdArray, int64_t, const NDArray&, bool);
template COOMatrix CSRRowWiseSamplingUniform2<kDGLCUDA, int64_t>(
  CSRMatrix, IdArray, int64_t, const NDArray&, bool);
template COOMatrix CSRRowWiseSamplingUniform3<kDGLCUDA, int32_t>(
  CSRMatrix, IdArray, int64_t, const NDArray&, const NDArray&, const NDArray&, bool);
template COOMatrix CSRRowWiseSamplingUniform3<kDGLCUDA, int64_t>(
  CSRMatrix, IdArray, int64_t, const NDArray&, const NDArray&, const NDArray&, bool);
template COOMatrix CSRRowWiseSamplingUniform4<kDGLCUDA, int32_t>(
  CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform4<kDGLCUDA, int64_t>(
  CSRMatrix, IdArray, int64_t, bool);


}  // namespace impl
}  // namespace aten
}  // namespace dgl
