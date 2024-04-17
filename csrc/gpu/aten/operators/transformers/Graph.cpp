#include "Graph.hpp"

namespace at {
namespace AtenIpexTypeXPU {
namespace onednn_graph {

// Thread local data-structures are required if multiple thread-pools
// of a PyTorch process would be used for inference.
thread_local std::unordered_map<std::bitset<32>, dnnl::graph::partition>
    partition_map_;
// Compiled partition (fused kernel) cache
// Adopted from
// https://github.com/lamerman/cpp-lru-cache/blob/master/include/lrucache.hpp

thread_local std::list<key_value_pair_t> cache_items_list_;
thread_local std::unordered_map<std::vector<int64_t>, list_iterator_t>
    fused_kernel_cache_map_;
// cache capacity is arbitrary
// TODO: Add an API to manipulate cache capacity
thread_local size_t capacity_ = 75000;

void insert_in_fused_kernel_cache(std::vector<int64_t>& map_key, cp_entry& cp) {
  cache_items_list_.push_front(key_value_pair_t(map_key, std::move(cp)));
  fused_kernel_cache_map_[map_key] = cache_items_list_.begin();
  if (fused_kernel_cache_map_.size() > capacity_) {
    auto last = cache_items_list_.end();
    last--;
    fused_kernel_cache_map_.erase(last->first);
    cache_items_list_.pop_back();
  }
}

void change_pos_in_list(list_iterator_t& kvpair) {
  cache_items_list_.splice(
      cache_items_list_.begin(), cache_items_list_, kvpair);
}

std::unordered_map<std::vector<int64_t>, list_iterator_t>::iterator cache_lookup(
    std::vector<int64_t>& map_key) {
  return fused_kernel_cache_map_.find(map_key);
}

std::unordered_map<std::vector<int64_t>, list_iterator_t>::iterator cache_end() {
  return fused_kernel_cache_map_.end();
}

std::unordered_map<std::bitset<32>, dnnl::graph::partition>::iterator
partition_map_lookup(std::bitset<32>& patternID) {
  return partition_map_.find(patternID);
}

std::unordered_map<std::bitset<32>, dnnl::graph::partition>::iterator
partition_map_end() {
  return partition_map_.end();
}

// The first 8 bits are reserved
// bit 0: is int8
// bit 1: is uint8
// bit 2: is fp16
// bit 3: is fp32
// bit 4: is sdp pattern
// bit 5: is MLP pattern
// bit 6: has conv. may or may not have linear as well
// bit 7: has linear, but is not an MLP
// The rest of the bits depend upon the arguments provided
// However, down the line, we might have different bitsets for different
// patterns
void insert_in_partition_cache(std::bitset<32>& patternID, partition& p) {
  partition_map_[patternID] = std::move(p);
}

void compile_partition(cp_entry& cp, const engine& eng) {
  cp.inputLogicalTensors_ = cp.partition_.get_input_ports();
  cp.outputLogicalTensors_ = cp.partition_.get_output_ports();
  cp.cp_ = cp.partition_.compile(
      cp.inputLogicalTensors_, cp.outputLogicalTensors_, eng);
}

} // end namespace onednn_graph
} // end namespace AtenIpexTypeXPU
} // end namespace at

