/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/fused_csc_sampling_graph.h
 * @brief Header file of csc sampling graph.
 */
#ifndef GRAPHBOLT_CSC_SAMPLING_GRAPH_H_
#define GRAPHBOLT_CSC_SAMPLING_GRAPH_H_

#include <graphbolt/fused_sampled_subgraph.h>
#include <graphbolt/shared_memory.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace graphbolt {
namespace sampling {

enum SamplerType { NEIGHBOR, LABOR };

template <SamplerType S>
struct SamplerArgs;

template <>
struct SamplerArgs<SamplerType::NEIGHBOR> {};

template <>
struct SamplerArgs<SamplerType::LABOR> {
  const torch::Tensor& indices;
  int64_t random_seed;
  int64_t num_nodes;
};

/**
 * @brief A sampling oriented csc format graph.
 *
 * Example usage:
 *
 * Suppose the graph has 3 node types, 3 edge types and 6 edges
 * auto node_type_offset = {0, 2, 4, 6}
 * auto type_per_edge = {0, 1, 0, 2, 1, 2}
 * auto graph = FusedCSCSamplingGraph(..., ..., node_type_offset, type_per_edge)
 *
 * The `node_type_offset` tensor represents the offset array of node type, the
 * given array indicates that node [0, 2) has type id 0, [2, 4) has type id 1,
 * and [4, 6) has type id 2. And the `type_per_edge` tensor represents the type
 * id of each edge.
 */
class FusedCSCSamplingGraph : public torch::CustomClassHolder {
 public:
  using NodeTypeToIDMap = torch::Dict<std::string, int64_t>;
  using EdgeTypeToIDMap = torch::Dict<std::string, int64_t>;
  using NodeAttrMap = torch::Dict<std::string, torch::Tensor>;
  using EdgeAttrMap = torch::Dict<std::string, torch::Tensor>;
  /** @brief Default constructor. */
  FusedCSCSamplingGraph() = default;

  /**
   * @brief Constructor for CSC with data.
   * @param indptr The CSC format index pointer array.
   * @param indices The CSC format index array.
   * @param node_type_offset A tensor representing the offset of node types, if
   * present.
   * @param type_per_edge A tensor representing the type of each edge, if
   * present.
   * @param node_type_to_id A dictionary mapping node type names to type IDs, if
   * present.
   * @param edge_type_to_id A dictionary mapping edge type names to type IDs, if
   * present.
   * @param node_attributes A dictionary of node attributes, if present.
   * @param edge_attributes A dictionary of edge attributes, if present.
   *
   */
  FusedCSCSamplingGraph(
      const torch::Tensor& indptr, const torch::Tensor& indices,
      const torch::optional<torch::Tensor>& node_type_offset = torch::nullopt,
      const torch::optional<torch::Tensor>& type_per_edge = torch::nullopt,
      const torch::optional<NodeTypeToIDMap>& node_type_to_id = torch::nullopt,
      const torch::optional<EdgeTypeToIDMap>& edge_type_to_id = torch::nullopt,
      const torch::optional<NodeAttrMap>& node_attributes = torch::nullopt,
      const torch::optional<EdgeAttrMap>& edge_attributes = torch::nullopt);

  /**
   * @brief Create a fused CSC graph from tensors of CSC format.
   * @param indptr Index pointer array of the CSC.
   * @param indices Indices array of the CSC.
   * @param node_type_offset A tensor representing the offset of node types, if
   * present.
   * @param type_per_edge A tensor representing the type of each edge, if
   * present.
   * @param node_type_to_id A dictionary mapping node type names to type IDs, if
   * present.
   * @param edge_type_to_id A dictionary mapping edge type names to type IDs, if
   * present.
   * @param node_attributes A dictionary of node attributes, if present.
   * @param edge_attributes A dictionary of edge attributes, if present.
   *
   * @return FusedCSCSamplingGraph
   */
  static c10::intrusive_ptr<FusedCSCSamplingGraph> Create(
      const torch::Tensor& indptr, const torch::Tensor& indices,
      const torch::optional<torch::Tensor>& node_type_offset,
      const torch::optional<torch::Tensor>& type_per_edge,
      const torch::optional<NodeTypeToIDMap>& node_type_to_id,
      const torch::optional<EdgeTypeToIDMap>& edge_type_to_id,
      const torch::optional<NodeAttrMap>& node_attributes,
      const torch::optional<EdgeAttrMap>& edge_attributes);

  /** @brief Get the number of nodes. */
  int64_t NumNodes() const { return indptr_.size(0) - 1; }

  /** @brief Get the number of edges. */
  int64_t NumEdges() const { return indices_.size(0); }

  /** @brief Get the csc index pointer tensor. */
  const torch::Tensor CSCIndptr() const { return indptr_; }

  /** @brief Get the index tensor. */
  const torch::Tensor Indices() const { return indices_; }

  /** @brief Get the node type offset tensor for a heterogeneous graph. */
  inline const torch::optional<torch::Tensor> NodeTypeOffset() const {
    return node_type_offset_;
  }

  /** @brief Get the edge type tensor for a heterogeneous graph. */
  inline const torch::optional<torch::Tensor> TypePerEdge() const {
    return type_per_edge_;
  }

  /**
   * @brief Get the node type to id map for a heterogeneous graph.
   * @note The map is a dictionary mapping node type names to type IDs.
   */
  inline const torch::optional<NodeTypeToIDMap> NodeTypeToID() const {
    return node_type_to_id_;
  }

  /**
   * @brief Get the edge type to id map for a heterogeneous graph.
   * @note The map is a dictionary mapping edge type names to type IDs.
   */
  inline const torch::optional<EdgeTypeToIDMap> EdgeTypeToID() const {
    return edge_type_to_id_;
  }

  /** @brief Get the node attributes dictionary. */
  inline const torch::optional<EdgeAttrMap> NodeAttributes() const {
    return node_attributes_;
  }

  /** @brief Get the edge attributes dictionary. */
  inline const torch::optional<EdgeAttrMap> EdgeAttributes() const {
    return edge_attributes_;
  }

  /**
   * @brief Get the node attribute tensor by name.
   *
   * If the input name is empty, return nullopt. Otherwise, return the node
   * attribute tensor by name.
   */
  inline torch::optional<torch::Tensor> NodeAttribute(
      torch::optional<std::string> name) const {
    if (!name.has_value()) {
      return torch::nullopt;
    }
    TORCH_CHECK(
        node_attributes_.has_value() &&
            node_attributes_.value().contains(name.value()),
        "Node attribute ", name.value(), " does not exist.");
    return torch::optional<torch::Tensor>(
        node_attributes_.value().at(name.value()));
  }

  /**
   * @brief Get the edge attribute tensor by name.
   *
   * If the input name is empty, return nullopt. Otherwise, return the edge
   * attribute tensor by name.
   */
  inline torch::optional<torch::Tensor> EdgeAttribute(
      torch::optional<std::string> name) const {
    if (!name.has_value()) {
      return torch::nullopt;
    }
    TORCH_CHECK(
        edge_attributes_.has_value() &&
            edge_attributes_.value().contains(name.value()),
        "Edge attribute ", name.value(), " does not exist.");
    return torch::optional<torch::Tensor>(
        edge_attributes_.value().at(name.value()));
  }

  /** @brief Set the csc index pointer tensor. */
  inline void SetCSCIndptr(const torch::Tensor& indptr) { indptr_ = indptr; }

  /** @brief Set the index tensor. */
  inline void SetIndices(const torch::Tensor& indices) { indices_ = indices; }

  /** @brief Set the node type offset tensor for a heterogeneous graph. */
  inline void SetNodeTypeOffset(
      const torch::optional<torch::Tensor>& node_type_offset) {
    node_type_offset_ = node_type_offset;
  }

  /** @brief Set the edge type tensor for a heterogeneous graph. */
  inline void SetTypePerEdge(
      const torch::optional<torch::Tensor>& type_per_edge) {
    type_per_edge_ = type_per_edge;
  }

  /**
   * @brief Set the node type to id map for a heterogeneous graph.
   * @note The map is a dictionary mapping node type names to type IDs.
   */
  inline void SetNodeTypeToID(
      const torch::optional<NodeTypeToIDMap>& node_type_to_id) {
    node_type_to_id_ = node_type_to_id;
  }

  /**
   * @brief Set the edge type to id map for a heterogeneous graph.
   * @note The map is a dictionary mapping edge type names to type IDs.
   */
  inline void SetEdgeTypeToID(
      const torch::optional<EdgeTypeToIDMap>& edge_type_to_id) {
    edge_type_to_id_ = edge_type_to_id;
  }

  /** @brief Set the node attributes dictionary. */
  inline void SetNodeAttributes(
      const torch::optional<EdgeAttrMap>& node_attributes) {
    node_attributes_ = node_attributes;
  }

  /** @brief Set the edge attributes dictionary. */
  inline void SetEdgeAttributes(
      const torch::optional<EdgeAttrMap>& edge_attributes) {
    edge_attributes_ = edge_attributes;
  }

  /**
   * @brief Magic number to indicate graph version in serialize/deserialize
   * stage.
   */
  static constexpr int64_t kCSCSamplingGraphSerializeMagic = 0xDD2E60F0F6B4A128;

  /**
   * @brief Load graph from stream.
   * @param archive Input stream for deserializing.
   */
  void Load(torch::serialize::InputArchive& archive);

  /**
   * @brief Save graph to stream.
   * @param archive Output stream for serializing.
   */
  void Save(torch::serialize::OutputArchive& archive) const;

  /**
   * @brief Pickle method for deserializing.
   * @param state The state of serialized FusedCSCSamplingGraph.
   */
  void SetState(
      const torch::Dict<std::string, torch::Dict<std::string, torch::Tensor>>&
          state);

  /**
   * @brief Pickle method for serializing.
   * @returns The state of this FusedCSCSamplingGraph.
   */
  torch::Dict<std::string, torch::Dict<std::string, torch::Tensor>> GetState()
      const;

  /**
   * @brief Return the subgraph induced on the inbound edges of the given nodes.
   * @param nodes Type agnostic node IDs to form the subgraph.
   *
   * @return FusedSampledSubgraph.
   */
  c10::intrusive_ptr<FusedSampledSubgraph> InSubgraph(
      const torch::Tensor& nodes) const;

  /**
   * @brief Sample neighboring edges of the given nodes and return the induced
   * subgraph.
   *
   * @param nodes The nodes from which to sample neighbors.
   * @param fanouts The number of edges to be sampled for each node with or
   * without considering edge types.
   *   - When the length is 1, it indicates that the fanout applies to all
   * neighbors of the node as a collective, regardless of the edge type.
   *   - Otherwise, the length should equal to the number of edge types, and
   * each fanout value corresponds to a specific edge type of the node.
   * The value of each fanout should be >= 0 or = -1.
   *   - When the value is -1, all neighbors will be chosen for sampling. It is
   * equivalent to selecting all neighbors with non-zero probability when the
   * fanout is >= the number of neighbors (and replacement is set to false).
   *   - When the value is a non-negative integer, it serves as a minimum
   * threshold for selecting neighbors.
   * @param replace Boolean indicating whether the sample is preformed with or
   * without replacement. If True, a value can be selected multiple times.
   * Otherwise, each value can be selected only once.
   * @param layer Boolean indicating whether neighbors should be sampled in a
   * layer sampling fashion. Uses the LABOR-0 algorithm to increase overlap of
   * sampled edges, see arXiv:2210.13339.
   * @param return_eids Boolean indicating whether edge IDs need to be returned,
   * typically used when edge features are required.
   * @param probs_name An optional string specifying the name of an edge
   * attribute. This attribute tensor should contain (unnormalized)
   * probabilities corresponding to each neighboring edge of a node. It must be
   * a 1D floating-point or boolean tensor, with the number of elements
   * equalling the total number of edges.
   *
   * @return An intrusive pointer to a FusedSampledSubgraph object containing
   * the sampled graph's information.
   */
  c10::intrusive_ptr<FusedSampledSubgraph> SampleNeighbors(
      const torch::Tensor& nodes, const std::vector<int64_t>& fanouts,
      bool replace, bool layer, bool return_eids,
      torch::optional<std::string> probs_name) const;

  /**
   * @brief Sample neighboring edges of the given nodes with a temporal
   * constraint. If `node_timestamp_attr_name` or `edge_timestamp_attr_name` is
   * given, the sampled neighbors or edges of an input node must have a
   * timestamp that is no later than that of the input node.
   *
   * @param nodes The nodes from which to sample neighbors.
   * @param input_nodes_timestamp The timestamp of the nodes.
   * @param fanouts The number of edges to be sampled for each node with or
   * without considering edge types, following the same rules as in
   * SampleNeighbors.
   * @param replace Boolean indicating whether the sample is preformed with or
   * without replacement. If True, a value can be selected multiple times.
   * Otherwise, each value can be selected only once.
   * @param return_eids Boolean indicating whether edge IDs need to be returned,
   * typically used when edge features are required.
   * @param probs_name An optional string specifying the name of an edge
   * attribute, following the same rules as in SampleNeighbors.
   * @param node_timestamp_attr_name An optional string specifying the name of
   * the node attribute that contains the timestamp of nodes in the graph.
   * @param edge_timestamp_attr_name An optional string specifying the name of
   * the edge attribute that contains the timestamp of edges in the graph.
   *
   * @return An intrusive pointer to a FusedSampledSubgraph object containing
   * the sampled graph's information.
   *
   */
  c10::intrusive_ptr<FusedSampledSubgraph> TemporalSampleNeighbors(
      const torch::Tensor& input_nodes,
      const torch::Tensor& input_nodes_timestamp,
      const std::vector<int64_t>& fanouts, bool replace, bool return_eids,
      torch::optional<std::string> probs_name,
      torch::optional<std::string> node_timestamp_attr_name,
      torch::optional<std::string> edge_timestamp_attr_name) const;

  /**
   * @brief Copy the graph to shared memory.
   * @param shared_memory_name The name of the shared memory.
   *
   * @return A new FusedCSCSamplingGraph object on shared memory.
   */
  c10::intrusive_ptr<FusedCSCSamplingGraph> CopyToSharedMemory(
      const std::string& shared_memory_name);

  /**
   * @brief Load the graph from shared memory.
   * @param shared_memory_name The name of the shared memory.
   *
   * @return A new FusedCSCSamplingGraph object on shared memory.
   */
  static c10::intrusive_ptr<FusedCSCSamplingGraph> LoadFromSharedMemory(
      const std::string& shared_memory_name);

  /**
   * @brief Hold the shared memory objects of the the tensor metadata and data.
   * @note Shared memory used to hold the tensor metadata and data of this
   * class. By storing its shared memory objects, the graph controls the
   * resources of shared memory, which will be released automatically when the
   * graph is destroyed. This function is for internal use by CopyToSharedMemory
   * and LoadFromSharedMemory. Please contact the DGL team if you need to use
   * it.
   * @param tensor_metadata_shm The shared memory objects of tensor metadata.
   * @param tensor_data_shm The shared memory objects of tensor data.
   */
  void HoldSharedMemoryObject(
      SharedMemoryPtr tensor_metadata_shm, SharedMemoryPtr tensor_data_shm);

 private:
  template <typename NumPickFn, typename PickFn>
  c10::intrusive_ptr<FusedSampledSubgraph> SampleNeighborsImpl(
      const torch::Tensor& nodes, bool return_eids, NumPickFn num_pick_fn,
      PickFn pick_fn) const;

  /** @brief CSC format index pointer array. */
  torch::Tensor indptr_;

  /** @brief CSC format index array. */
  torch::Tensor indices_;

  /**
   * @brief Offset array of node type. The length of it is equal to the number
   * of node types + 1. The tensor is in ascending order as nodes of the same
   * type have continuous IDs, and larger node IDs are paired with larger node
   * type IDs. Its first value is 0 and last value is the number of nodes. And
   * nodes with ID between `node_type_offset_[i] ~ node_type_offset_[i+1]` are
   * of type id `i`.
   */
  torch::optional<torch::Tensor> node_type_offset_;

  /**
   * @brief Type id of each edge, where type id is the corresponding index of
   * edge types. The length of it is equal to the number of edges.
   */
  torch::optional<torch::Tensor> type_per_edge_;

  /**
   * @brief A dictionary mapping node type names to type IDs. The length of it
   * is equal to the number of node types. The key is the node type name, and
   * the value is the corresponding type ID.
   */
  torch::optional<NodeTypeToIDMap> node_type_to_id_;

  /**
   * @brief A dictionary mapping edge type names to type IDs. The length of it
   * is equal to the number of edge types. The key is the edge type name, and
   * the value is the corresponding type ID.
   */
  torch::optional<EdgeTypeToIDMap> edge_type_to_id_;

  /**
   * @brief A dictionary of node attributes. Each key represents the attribute's
   * name, while the corresponding value holds the attribute's specific value.
   * The length of each value should match the total number of nodes."
   */
  torch::optional<NodeAttrMap> node_attributes_;

  /**
   * @brief A dictionary of edge attributes. Each key represents the attribute's
   * name, while the corresponding value holds the attribute's specific value.
   * The length of each value should match the total number of edges."
   */
  torch::optional<EdgeAttrMap> edge_attributes_;

  /**
   * @brief Shared memory used to hold the tensor metadata and data of this
   * class. By storing its shared memory objects, the graph controls the
   * resources of shared memory, which will be released automatically when the
   * graph is destroyed.
   */
  SharedMemoryPtr tensor_metadata_shm_, tensor_data_shm_;
};

/**
 * @brief Calculate the number of the neighbors to be picked for the given node.
 *
 * @param fanout The number of edges to be sampled for each node. It should be
 * >= 0 or -1.
 *  - When the value is -1, all neighbors (with non-zero probability, if
 * weighted) will be chosen for sampling. It is equivalent to selecting all
 * neighbors with non-zero probability when the fanout is >= the number of
 * neighbors (and replacement is set to false).
 *  - When the value is a non-negative integer, it serves as a minimum
 * threshold for selecting neighbors.
 * @param replace Boolean indicating whether the sample is performed with or
 * without replacement. If True, a value can be selected multiple times.
 * Otherwise, each value can be selected only once.
 * @param probs_or_mask Optional tensor containing the (unnormalized)
 * probabilities associated with each neighboring edge of a node in the original
 * graph. It must be a 1D floating-point tensor with the number of elements
 * equal to the number of edges in the graph.
 * @param offset The starting edge ID for the connected neighbors of the given
 * node.
 * @param num_neighbors The number of neighbors of this node.
 *
 * @return The pick number of the given node.
 */
int64_t NumPick(
    int64_t fanout, bool replace,
    const torch::optional<torch::Tensor>& probs_or_mask, int64_t offset,
    int64_t num_neighbors);

int64_t TemporalNumPick(
    torch::Tensor seed_timestamp, torch::Tensor csc_indics, int64_t fanout,
    bool replace, const torch::optional<torch::Tensor>& probs_or_mask,
    const torch::optional<torch::Tensor>& node_timestamp,
    const torch::optional<torch::Tensor>& edge_timestamp, int64_t seed_offset,
    int64_t offset, int64_t num_neighbors);

int64_t NumPickByEtype(
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::Tensor& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask, int64_t offset,
    int64_t num_neighbors);

int64_t TemporalNumPickByEtype(
    torch::Tensor seed_timestamp, torch::Tensor csc_indices,
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::Tensor& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask,
    const torch::optional<torch::Tensor>& node_timestamp,
    const torch::optional<torch::Tensor>& edge_timestamp, int64_t seed_offset,
    int64_t offset, int64_t num_neighbors);

/**
 * @brief Picks a specified number of neighbors for a node, starting from the
 * given offset and having the specified number of neighbors.
 *
 * If 'probs_or_mask' is provided, it indicates that the sampling is
 * non-uniform. In such cases:
 * - When the number of neighbors with non-zero probability is less than or
 * equal to fanout, all neighbors with non-zero probability will be selected.
 * - When the number of neighbors with non-zero probability exceeds fanout, the
 * sampling process will select 'fanout' elements based on their respective
 * probabilities. Higher probabilities will increase the chances of being chosen
 * during the sampling process.
 *
 * @param offset The starting edge ID for the connected neighbors of the sampled
 * node.
 * @param num_neighbors The number of neighbors to pick.
 * @param fanout The number of edges to be sampled for each node. It should be
 * >= 0 or -1.
 *  - When the value is -1, all neighbors will be chosen for sampling. It is
 * equivalent to selecting all neighbors with non-zero probability when the
 * fanout is >= the number of neighbors (and replacement is set to false).
 *  - When the value is a non-negative integer, it serves as a minimum
 * threshold for selecting neighbors.
 * @param replace Boolean indicating whether the sample is preformed with or
 * without replacement. If True, a value can be selected multiple times.
 * Otherwise, each value can be selected only once.
 * @param options Tensor options specifying the desired data type of the result.
 * @param probs_or_mask Optional tensor containing the (unnormalized)
 * probabilities associated with each neighboring edge of a node in the original
 * graph. It must be a 1D floating-point tensor with the number of elements
 * equal to the number of edges in the graph.
 * @param picked_data_ptr The destination address where the picked neighbors
 * should be put. Enough memory space should be allocated in advance.
 */
template <typename PickedType>
int64_t Pick(
    int64_t offset, int64_t num_neighbors, int64_t fanout, bool replace,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask,
    SamplerArgs<SamplerType::NEIGHBOR> args, PickedType* picked_data_ptr);

template <typename PickedType>
int64_t Pick(
    int64_t offset, int64_t num_neighbors, int64_t fanout, bool replace,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask,
    SamplerArgs<SamplerType::LABOR> args, PickedType* picked_data_ptr);

template <typename PickedType>
int64_t TemporalPick(
    torch::Tensor seed_timestamp, torch::Tensor csc_indices,
    int64_t seed_offset, int64_t offset, int64_t num_neighbors, int64_t fanout,
    bool replace, const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask,
    const torch::optional<torch::Tensor>& node_timestamp,
    const torch::optional<torch::Tensor>& edge_timestamp,
    PickedType* picked_data_ptr);

/**
 * @brief Picks a specified number of neighbors for a node per edge type,
 * starting from the given offset and having the specified number of neighbors.
 *
 * @param offset The starting edge ID for the connected neighbors of the sampled
 * node.
 * @param num_neighbors The number of neighbors to pick.
 * @param fanouts The edge sampling numbers corresponding to each edge type for
 * a single node. The value of each fanout should be >= 0 or = 1.
 *  - When the value is -1, all neighbors with non-zero probability will be
 * chosen for sampling. It is equivalent to selecting all neighbors when the
 * fanout is >= the number of neighbors (and replacement is set to false).
 *  - When the value is a non-negative integer, it serves as a minimum threshold
 * for selecting neighbors.
 * @param replace Boolean indicating whether the sample is preformed with or
 * without replacement. If True, a value can be selected multiple times.
 * Otherwise, each value can be selected only once.
 * @param options Tensor options specifying the desired data type of the result.
 * @param type_per_edge Tensor representing the type of each edge in the
 * original graph.
 * @param probs_or_mask Optional tensor containing the (unnormalized)
 * probabilities associated with each neighboring edge of a node in the original
 * graph. It must be a 1D floating-point tensor with the number of elements
 * equal to the number of edges in the graph.
 * @param picked_data_ptr The destination address where the picked neighbors
 * should be put. Enough memory space should be allocated in advance.
 */
template <SamplerType S, typename PickedType>
int64_t PickByEtype(
    int64_t offset, int64_t num_neighbors, const std::vector<int64_t>& fanouts,
    bool replace, const torch::TensorOptions& options,
    const torch::Tensor& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask, SamplerArgs<S> args,
    PickedType* picked_data_ptr);

template <typename PickedType>
int64_t TemporalPickByEtype(
    torch::Tensor seed_timestamp, torch::Tensor csc_indices,
    int64_t seed_offset, int64_t offset, int64_t num_neighbors,
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::TensorOptions& options, const torch::Tensor& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask,
    const torch::optional<torch::Tensor>& node_timestamp,
    const torch::optional<torch::Tensor>& edge_timestamp,
    PickedType* picked_data_ptr);

template <
    bool NonUniform, bool Replace, typename ProbsType, typename PickedType,
    int StackSize = 1024>
int64_t LaborPick(
    int64_t offset, int64_t num_neighbors, int64_t fanout,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask,
    SamplerArgs<SamplerType::LABOR> args, PickedType* picked_data_ptr);

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_CSC_SAMPLING_GRAPH_H_
