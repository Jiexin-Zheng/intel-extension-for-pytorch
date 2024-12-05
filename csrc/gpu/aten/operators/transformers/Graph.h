#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <torch/library.h>
#include <bitset>
#include <list>

namespace at::AtenIpexTypeXPU::onednn::graph {

using namespace dnnl::graph;

using stream = dnnl::stream;
using logical_tensor = dnnl::graph::logical_tensor;
using op = dnnl::graph::op;
using graph = dnnl::graph::graph;
using partition = dnnl::graph::partition;
using compiled_partition = dnnl::graph::compiled_partition;
using tensor = dnnl::graph::tensor;

using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;
using engine = dnnl::engine;

// Compiled Partition entry
struct cp_entry {
  partition partition_;
  std::vector<logical_tensor> input_logical_tensors;
  std::vector<logical_tensor> output_logical_tensors;
  compiled_partition cp;
};
struct GraphCache {
  using key_value_pair_t = std::pair<std::vector<int64_t>, cp_entry>;
  using list_iterator_t = std::list<key_value_pair_t>::iterator;

  struct CompiledPartitionKeyHasher {
    size_t operator()(const std::vector<int64_t>& key) const {
      size_t acc = 0;
      std::hash<int64_t> hasher;
      acc = c10::hash_combine(acc, hasher(key.size()));
      for (size_t i = 0; i < key.size(); i++) {
        acc = c10::hash_combine(acc, hasher(key[i]));
      }
      return acc;
    }
  };

  std::unordered_map<std::bitset<32>, dnnl::graph::partition> partition_map_;
  std::list<key_value_pair_t> cache_items_list_;
  std::unordered_map<
      std::vector<int64_t>,
      list_iterator_t,
      CompiledPartitionKeyHasher>
      fused_kernel_map_;
  size_t capacity_ = 1024;

  cp_entry& insert_fused_kernel_cache(
      std::vector<int64_t>& map_key,
      cp_entry& cp) {
    cache_items_list_.push_front(key_value_pair_t(map_key, std::move(cp)));
    fused_kernel_map_[map_key] = cache_items_list_.begin();
    if (fused_kernel_map_.size() > capacity_) {
      auto last = cache_items_list_.end();
      last--;
      fused_kernel_map_.erase(last->first);
      cache_items_list_.pop_back();
    }
    return fused_kernel_map_[map_key]->second;
  }
  // The first 8 bits are reserved
  // bit 0: is int8
  // bit 1: is uint8
  // bit 2: is fp16
  // bit 3: is fp32
  // bit 4: is sdp pattern
  // bit 5-7: N/A
  // The rest of the bits depend upon the arguments provided
  // However, down the line, we might have different bitsets for different
  // patterns
  partition& insert_partition_cache(std::bitset<32>& patternID, partition& p) {
    partition_map_[patternID] = std::move(p);
    return partition_map_[patternID];
  }
  std::optional<std::reference_wrapper<dnnl::graph::partition>> find_partition(
      std::bitset<32>& patternID) {
    auto iter = partition_map_.find(patternID);
    if (iter != partition_map_.end()) {
      return iter->second;
    }
    return std::nullopt;
  }
  std::optional<std::reference_wrapper<cp_entry>> find_kernel(
      std::vector<int64_t>& map_key) {
    auto iter = fused_kernel_map_.find(map_key);
    if (iter != fused_kernel_map_.end()) {
      return iter->second->second;
    }
    return std::nullopt;
  }
};

void gpu_float_sdpa(
    int batch_size,
    int seq_len_q,
    int seq_len_k,
    int num_head,
    int num_head_kv,
    int head_dim,
    int head_dim_v,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    bool enable_gqa,
    bool is_causal,
    float softmax_scale,
    const Tensor& output);
} // namespace at::AtenIpexTypeXPU::onednn::graph
