/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/index_select_csc_impl.cu
 * @brief Index select csc operator implementation on CUDA.
 */
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <graphbolt/cuda_ops.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <numeric>

#include "./common.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

constexpr int BLOCK_SIZE = 128;

// Given the in_degree array and a permutation, returns in_degree of the output
// and the permuted and modified in_degree of the input. The modified in_degree
// is modified so that there is slack to be able to align as needed.
template <typename indptr_t, typename indices_t>
struct AlignmentFunc {
  static_assert(GPU_CACHE_LINE_SIZE % sizeof(indices_t) == 0);
  const indptr_t* in_degree;
  const int64_t* perm;
  int64_t num_nodes;
  __host__ __device__ auto operator()(int64_t row) {
    constexpr int num_elements = GPU_CACHE_LINE_SIZE / sizeof(indices_t);
    return thrust::make_tuple(
        in_degree[row],
        // A single cache line has num_elements items, we add num_elements - 1
        // to ensure there is enough slack to move forward or backward by
        // num_elements - 1 items if the performed access is not aligned.
        (indptr_t)(in_degree[perm ? perm[row % num_nodes] : row] + num_elements - 1));
  }
};

template <typename indptr_t, typename indices_t>
__global__ void _CopyIndicesAlignedKernel(
    const indptr_t edge_count, const int64_t num_nodes,
    const indptr_t* const indptr, const indptr_t* const output_indptr,
    const indptr_t* const output_indptr_aligned, const indices_t* const indices,
    indices_t* const output_indices, const int64_t* const perm) {
  indptr_t idx = static_cast<indptr_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;

  while (idx < edge_count) {
    const auto permuted_row_pos =
        cuda::UpperBound(output_indptr_aligned, num_nodes, idx) - 1;
    const auto row_pos = perm ? perm[permuted_row_pos] : permuted_row_pos;
    const auto out_row = output_indptr[row_pos];
    const auto d = output_indptr[row_pos + 1] - out_row;
    const int offset =
        ((size_t)(indices + indptr[row_pos] - output_indptr_aligned[permuted_row_pos]) %
         GPU_CACHE_LINE_SIZE) /
        sizeof(indices_t);
    const auto rofs = idx - output_indptr_aligned[permuted_row_pos] - offset;
    if (rofs >= 0 && rofs < d) {
      const auto in_idx = indptr[row_pos] + rofs;
      assert((size_t)(indices + in_idx - idx) % GPU_CACHE_LINE_SIZE == 0);
      const auto u = indices[in_idx];
      output_indices[out_row + rofs] = u;
    }
    idx += stride_x;
  }
}

// Given rows and indptr, computes:
// inrow_indptr[i] = indptr[rows[i]];
// in_degree[i] = indptr[rows[i] + 1] - indptr[rows[i]];
template <typename indptr_t, typename nodes_t>
struct SliceFunc {
  const nodes_t* rows;
  const indptr_t* indptr;
  indptr_t* in_degree;
  indptr_t* inrow_indptr;
  __host__ __device__ auto operator()(int64_t tIdx) {
    const auto out_row = rows[tIdx];
    const auto indptr_val = indptr[out_row];
    const auto degree = indptr[out_row + 1] - indptr_val;
    in_degree[tIdx] = degree;
    inrow_indptr[tIdx] = indptr_val;
  }
};

struct PairSum {
  template <typename indptr_t>
  __host__ __device__ auto operator()(
      const thrust::tuple<indptr_t, indptr_t> a,
      const thrust::tuple<indptr_t, indptr_t> b) {
    return thrust::make_tuple(
        thrust::get<0>(a) + thrust::get<0>(b),
        thrust::get<1>(a) + thrust::get<1>(b));
  };
};

// Returns (indptr[nodes + 1] - indptr[nodes], indptr[nodes])
std::tuple<torch::Tensor, torch::Tensor> SliceCSCIndptr(
    torch::Tensor indptr, torch::Tensor nodes) {
  auto allocator = cuda::GetAllocator();
  const auto exec_policy =
      thrust::cuda::par_nosync(allocator).on(cuda::GetCurrentStream());
  const int64_t num_nodes = nodes.size(0);
  // Read indptr only once in case it is pinned and access is slow.
  auto sliced_indptr =
      torch::empty(num_nodes, nodes.options().dtype(indptr.scalar_type()));
  // compute in-degrees
  auto in_degree =
      torch::empty(num_nodes + 1, nodes.options().dtype(indptr.scalar_type()));
  thrust::counting_iterator<int64_t> iota(0);
  AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "IndexSelectCSCIndptr", ([&] {
        using indptr_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(
            nodes.scalar_type(), "IndexSelectCSCNodes", ([&] {
              using nodes_t = index_t;
              thrust::for_each(
                  exec_policy, iota, iota + num_nodes,
                  SliceFunc<indptr_t, nodes_t>{
                      nodes.data_ptr<nodes_t>(), indptr.data_ptr<indptr_t>(),
                      in_degree.data_ptr<indptr_t>(),
                      sliced_indptr.data_ptr<indptr_t>()});
            }));
      }));
  return {in_degree, sliced_indptr};
}

template <typename indptr_t, typename indices_t>
std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSCCopyIndices(
    torch::Tensor indices, const int64_t num_nodes,
    const indptr_t* const in_degree, const indptr_t* const sliced_indptr,
    const int64_t* const perm, torch::TensorOptions nodes_options,
    torch::ScalarType indptr_scalar_type, cudaStream_t stream) {
  auto allocator = cuda::GetAllocator();
  thrust::counting_iterator<int64_t> iota(0);

  // Output indptr for the slice indexed by nodes.
  auto output_indptr =
      torch::empty(num_nodes + 1, nodes_options.dtype(indptr_scalar_type));

  auto output_indptr_aligned =
      allocator.AllocateStorage<indptr_t>(num_nodes + 1);

  {
    // Returns the actual and modified_indegree as a pair, the
    // latter overestimates the actual indegree for alignment
    // purposes.
    auto modified_in_degree = thrust::make_transform_iterator(
        iota, AlignmentFunc<indptr_t, indices_t>{in_degree, perm, num_nodes});
    auto output_indptr_pair = thrust::make_zip_iterator(
        output_indptr.data_ptr<indptr_t>(), output_indptr_aligned.get());
    thrust::tuple<indptr_t, indptr_t> zero_value{};
    // Compute the prefix sum over actual and modified indegrees.
    size_t tmp_storage_size = 0;
    CUDA_CALL(cub::DeviceScan::ExclusiveScan(
        nullptr, tmp_storage_size, modified_in_degree, output_indptr_pair,
        PairSum{}, zero_value, num_nodes + 1, stream));
    auto tmp_storage = allocator.AllocateStorage<char>(tmp_storage_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveScan(
        tmp_storage.get(), tmp_storage_size, modified_in_degree,
        output_indptr_pair, PairSum{}, zero_value, num_nodes + 1, stream));
  }

  // Copy the actual total number of edges.
  auto edge_count =
      cuda::CopyScalar{output_indptr.data_ptr<indptr_t>() + num_nodes};
  // Copy the modified number of edges.
  auto edge_count_aligned =
      cuda::CopyScalar{output_indptr_aligned.get() + num_nodes};

  // Allocate output array with actual number of edges.
  torch::Tensor output_indices = torch::empty(
      static_cast<indptr_t>(edge_count),
      nodes_options.dtype(indices.scalar_type()));
  const dim3 block(BLOCK_SIZE);
  const dim3 grid(
      (static_cast<indptr_t>(edge_count_aligned) + BLOCK_SIZE - 1) /
      BLOCK_SIZE);

  // Perform the actual copying, of the indices array into
  // output_indices in an aligned manner.
  CUDA_KERNEL_CALL(
      _CopyIndicesAlignedKernel, grid, block, 0, stream,
      static_cast<indptr_t>(edge_count_aligned), num_nodes, sliced_indptr,
      output_indptr.data_ptr<indptr_t>(), output_indptr_aligned.get(),
      reinterpret_cast<indices_t*>(indices.data_ptr()),
      reinterpret_cast<indices_t*>(output_indices.data_ptr()), perm);
  return {output_indptr, output_indices};
}

std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes) {
  // Sorting nodes so that accesses over PCI-e are more regular.
  const auto sorted_idx =
      Sort(nodes, cuda::NumberOfBits(indptr.size(0) - 1)).second;
  auto stream = cuda::GetCurrentStream();
  const int64_t num_nodes = nodes.size(0);

  auto in_degree_and_sliced_indptr = SliceCSCIndptr(indptr, nodes);
  return AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "UVAIndexSelectCSCIndptr", ([&] {
        using indptr_t = scalar_t;
        auto in_degree =
            std::get<0>(in_degree_and_sliced_indptr).data_ptr<indptr_t>();
        auto sliced_indptr =
            std::get<1>(in_degree_and_sliced_indptr).data_ptr<indptr_t>();
        return GRAPHBOLT_DISPATCH_ELEMENT_SIZES(
            indices.element_size(), "UVAIndexSelectCSCCopyIndices", ([&] {
              return UVAIndexSelectCSCCopyIndices<indptr_t, element_size_t>(
                  indices, num_nodes, in_degree, sliced_indptr,
                  sorted_idx.data_ptr<int64_t>(), nodes.options(),
                  indptr.scalar_type(), stream);
            }));
      }));
}

template <typename indptr_t, typename indices_t>
struct IteratorFunc {
  indptr_t* indptr;
  indices_t* indices;
  __host__ __device__ auto operator()(int64_t i) { return indices + indptr[i]; }
};

template <typename indptr_t, typename indices_t>
struct ConvertToBytes {
  const indptr_t* in_degree;
  __host__ __device__ indptr_t operator()(int64_t i) {
    return in_degree[i] * sizeof(indices_t);
  }
};

template <typename indptr_t, typename indices_t>
void IndexSelectCSCCopyIndices(
    const int64_t num_nodes, indices_t* const indices,
    indptr_t* const sliced_indptr, const indptr_t* const in_degree,
    indptr_t* const output_indptr, indices_t* const output_indices,
    cudaStream_t stream) {
  auto allocator = cuda::GetAllocator();
  thrust::counting_iterator<int64_t> iota(0);

  auto input_buffer_it = thrust::make_transform_iterator(
      iota, IteratorFunc<indptr_t, indices_t>{sliced_indptr, indices});
  auto output_buffer_it = thrust::make_transform_iterator(
      iota, IteratorFunc<indptr_t, indices_t>{output_indptr, output_indices});
  auto buffer_sizes = thrust::make_transform_iterator(
      iota, ConvertToBytes<indptr_t, indices_t>{in_degree});
  constexpr int64_t max_copy_at_once = std::numeric_limits<int32_t>::max();

  // Performs the copy from indices into output_indices.
  for (int64_t i = 0; i < num_nodes; i += max_copy_at_once) {
    size_t tmp_storage_size = 0;
    CUDA_CALL(cub::DeviceMemcpy::Batched(
        nullptr, tmp_storage_size, input_buffer_it + i, output_buffer_it + i,
        buffer_sizes + i, std::min(num_nodes - i, max_copy_at_once), stream));
    auto tmp_storage = allocator.AllocateStorage<char>(tmp_storage_size);
    CUDA_CALL(cub::DeviceMemcpy::Batched(
        tmp_storage.get(), tmp_storage_size, input_buffer_it + i,
        output_buffer_it + i, buffer_sizes + i,
        std::min(num_nodes - i, max_copy_at_once), stream));
  }
}

std::tuple<torch::Tensor, torch::Tensor> DeviceIndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes) {
  auto stream = cuda::GetCurrentStream();
  const int64_t num_nodes = nodes.size(0);
  auto in_degree_and_sliced_indptr = SliceCSCIndptr(indptr, nodes);
  return AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "IndexSelectCSCIndptr", ([&] {
        using indptr_t = scalar_t;
        auto in_degree =
            std::get<0>(in_degree_and_sliced_indptr).data_ptr<indptr_t>();
        auto sliced_indptr =
            std::get<1>(in_degree_and_sliced_indptr).data_ptr<indptr_t>();
        // Output indptr for the slice indexed by nodes.
        torch::Tensor output_indptr = torch::empty(
            num_nodes + 1, nodes.options().dtype(indptr.scalar_type()));

        {  // Compute the output indptr, output_indptr.
          size_t tmp_storage_size = 0;
          CUDA_CALL(cub::DeviceScan::ExclusiveSum(
              nullptr, tmp_storage_size, in_degree,
              output_indptr.data_ptr<indptr_t>(), num_nodes + 1, stream));
          auto allocator = cuda::GetAllocator();
          auto tmp_storage = allocator.AllocateStorage<char>(tmp_storage_size);
          CUDA_CALL(cub::DeviceScan::ExclusiveSum(
              tmp_storage.get(), tmp_storage_size, in_degree,
              output_indptr.data_ptr<indptr_t>(), num_nodes + 1, stream));
        }

        // Number of edges being copied.
        auto edge_count =
            cuda::CopyScalar{output_indptr.data_ptr<indptr_t>() + num_nodes};
        // Allocate output array of size number of copied edges.
        torch::Tensor output_indices = torch::empty(
            static_cast<indptr_t>(edge_count),
            nodes.options().dtype(indices.scalar_type()));
        GRAPHBOLT_DISPATCH_ELEMENT_SIZES(
            indices.element_size(), "IndexSelectCSCCopyIndices", ([&] {
              using indices_t = element_size_t;
              IndexSelectCSCCopyIndices<indptr_t, indices_t>(
                  num_nodes, reinterpret_cast<indices_t*>(indices.data_ptr()),
                  sliced_indptr, in_degree, output_indptr.data_ptr<indptr_t>(),
                  reinterpret_cast<indices_t*>(output_indices.data_ptr()),
                  stream);
            }));
        return std::make_tuple(output_indptr, output_indices);
      }));
}

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes) {
  if (indices.is_pinned()) {
    return UVAIndexSelectCSCImpl(indptr, indices, nodes);
  } else {
    return DeviceIndexSelectCSCImpl(indptr, indices, nodes);
  }
}

}  //  namespace ops
}  //  namespace graphbolt
