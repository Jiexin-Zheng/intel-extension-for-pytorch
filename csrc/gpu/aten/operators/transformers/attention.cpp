#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "../Blas.h"
#include "../comm/ATDispatch.h"
#include "sdp_utils.h"
#include "utils/CustomOperatorRegistration.h"

#include <omp.h>
#include "../xetla/mha.h"
#include "Graph.hpp"
#include "aten/core/Device.h"
#include "oneDNN/Runtime.h"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"

namespace at {
namespace AtenIpexTypeXPU {

using namespace onednn_graph;

using stream = dnnl::stream;
using logical_tensor = dnnl::graph::logical_tensor;
using op = dnnl::graph::op;
using graph = dnnl::graph::graph;
using partition = dnnl::graph::partition;
using compiled_partition = dnnl::graph::compiled_partition;
using tensor = dnnl::graph::tensor;

void allocate_sycl_graph_mem(
    std::vector<tensor>& tensors,
    const logical_tensor& lt,
    const engine& eng,
    const Tensor& input) {
  tensor new_ts{lt, eng, input.data_ptr()};
  tensors.push_back(new_ts);
}

// (TODO:Jiexin)Refine the cache part.
void gpu_float_sdpa(
    int batch_size,
    int seq_len_q,
    int seq_len_k,
    int num_head,
    int size_per_head,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask,
    const float& softmax_scale,
    const Tensor& output,
    bool query_requires_transpose = false,
    bool key_requires_transpose_twice = false,
    bool key_requires_transpose_once = false,
    bool value_requires_transpose = false,
    bool apply_mask_before_scale = false,
    bool choose_causal_mask_over_attn_score = false,
    bool output_requires_transpose_and_reorder = false) {
  engine::kind ekind = engine::kind::gpu;
  auto eng = xpu::oneDNN::GpuEngineManager::Instance().get_engine(
      {kXPU, current_device()});
  auto strm = xpu::oneDNN::GpuStreamManager::Instance().get_stream();

  data_type dtype = data_type::undef;

  Tensor softmax_scale1 = full(
      {},
      1 / softmax_scale,
      TensorOptions().dtype(c10::ScalarType::Half).device(DeviceType::XPU));

  // prepare cache key params
  //(TODO:Jiexin) need to support more general sdp patterns
  query_requires_transpose = false;
  key_requires_transpose_twice = false;
  key_requires_transpose_once = true;
  value_requires_transpose = false;
  apply_mask_before_scale = false;
  choose_causal_mask_over_attn_score = false;
  output_requires_transpose_and_reorder = true;

  // cache key creation
  // patternID is determined on the basis of the arguments provided
  std::bitset<32> patternID;
  if (query.scalar_type() == c10::ScalarType::Float) {
    // bit 3 corresponds to float32 dtype
    patternID.set(3, 1);
    dtype = data_type::f32;
  } else {
    // bit 2 corresponds to float16 dtype
    patternID.set(2, 1);
    dtype = data_type::f16;
  }
  // sdp pattern
  patternID.set(4, 1);
  // Refer to comments in Graph.cpp. The first 8 bits are reserved
  int pos = 8;

  patternID.set(pos++, query_requires_transpose);
  patternID.set(pos++, key_requires_transpose_twice);
  patternID.set(pos++, key_requires_transpose_once);
  patternID.set(pos++, value_requires_transpose);
  patternID.set(pos++, apply_mask_before_scale);
  patternID.set(pos++, choose_causal_mask_over_attn_score);
  patternID.set(pos++, output_requires_transpose_and_reorder);
  // first check cache
  // The key has a pattern ID, as well as the shapes of input tenors
  std::vector<int64_t> map_key;
  map_key.reserve(1024);
  // We use this because different thread-pools may be used
  map_key.push_back(omp_get_max_threads());

  map_key.push_back(static_cast<int64_t>(patternID.to_ullong()));

  map_key.insert(map_key.end(), key.sizes().begin(), key.sizes().end());
  map_key.insert(map_key.end(), query.sizes().begin(), query.sizes().end());
  map_key.insert(map_key.end(), value.sizes().begin(), value.sizes().end());
  map_key.insert(map_key.end(), attn_mask.sizes().begin(), attn_mask.sizes().end());
  map_key.insert(
      map_key.end(), softmax_scale1.sizes().begin(), softmax_scale1.sizes().end());
  auto iter = cache_lookup(map_key);
  if (iter == cache_end()) {
    // compiled partition cache no hit
    cp_entry compiledPartitionEntry;
    auto graph_partition_iter = partition_map_lookup(patternID);
    partition graph_partition;
    if (graph_partition_iter == partition_map_end()) {
      // partition cache no hit
      TORCH_CHECK(
          ((dtype == data_type::f16) || (dtype == data_type::f32)),
          "Only F16 & FP32 datatypes are currently supported");

      // graph building and partitioning
      // currently, we assume that Q and K have same sequence length
      int seq_len = seq_len_q;

      int head_dim = size_per_head * num_head;
      dims qkv_input_shape = {batch_size, num_head, seq_len, size_per_head};
      dims qk_output_shape = {batch_size, num_head, seq_len, seq_len};
      dims scale_shape = {1};
      dims attention_mask_shape = {attn_mask.sizes().vec()};
      dims qkv_transpose_order = {0, 1, 2, 3};
      dims qkv_transposed_shape = {
          batch_size, num_head, seq_len, size_per_head};
      dims qkv_reshaped_shape = {batch_size, num_head, seq_len, size_per_head};

      size_t lt_id = 0;

      logical_tensor query_input{
          lt_id++, dtype, qkv_input_shape, query.strides().vec()};
      logical_tensor key_input{
          lt_id++, dtype, qkv_input_shape, key.strides().vec()};

      logical_tensor matmul_qk_out{
          lt_id++,
          dtype,
          qk_output_shape,
          logical_tensor::layout_type::strided};
      op matmul_qk{
          0,
          op::kind::MatMul,
          {query_input, key_input},
          {matmul_qk_out},
          "matmul_qk"};
      matmul_qk.set_attr<bool>(op::attr::transpose_b, true);

      logical_tensor scale_factor{
          lt_id++,
          dtype,
          scale_shape,
          logical_tensor::layout_type::strided,
          logical_tensor::property_type::constant};
      logical_tensor scaled_qk_out{
          lt_id++,
          dtype,
          qk_output_shape,
          logical_tensor::layout_type::strided};
      op scale_div{
          1,
          op::kind::Divide,
          {matmul_qk_out, scale_factor},
          {scaled_qk_out},
          "scale_div"};
    logical_tensor attention_mask{
         lt_id++, dtype, attention_mask_shape, attn_mask.strides().vec()};
      logical_tensor masked_qk_out{
          lt_id++,
          dtype,
          qk_output_shape,
          logical_tensor::layout_type::strided};
      op mask_add{
          2,
          op::kind::Add,
          {scaled_qk_out, attention_mask},
          {masked_qk_out},
          "mask_add"};

      logical_tensor softmax_out{
          lt_id++,
          dtype,
          qk_output_shape,
          logical_tensor::layout_type::strided};
      op softmax{
          3, op::kind::SoftMax, {masked_qk_out}, {softmax_out}, "softmax"};
      softmax.set_attr<int64_t>(op::attr::axis, -1);

      logical_tensor value_input{
          lt_id++, dtype, qkv_input_shape, value.strides().vec()};
      logical_tensor matmul_v_out{
          lt_id++, dtype, qkv_input_shape, output.strides().vec()};

      op matmul_v{
          4,
          op::kind::MatMul,
          {softmax_out, value_input},
          {matmul_v_out},
          "matmul_v"};

      logical_tensor qkv_transposed_out{
          lt_id++,
          dtype,
          qkv_transposed_shape,
          logical_tensor::layout_type::strided};
      op transpose{
          5,
          op::kind::StaticTranspose,
          {matmul_v_out},
          {qkv_transposed_out},
          "transpose"};
      transpose.set_attr<std::vector<int64_t>>(
          op::attr::order, qkv_transpose_order);

      logical_tensor qkv_reshaped_out{
          lt_id++, dtype, qkv_reshaped_shape, output.strides().vec()};
      op reshape{
          6,
          op::kind::StaticReshape,
          {qkv_transposed_out},
          {qkv_reshaped_out},
          "reshape"};
      reshape.set_attr(op::attr::special_zero, false);
      reshape.set_attr<std::vector<int64_t>>(
          op::attr::shape, qkv_reshaped_shape);

      graph g(ekind);
      g.add_op(matmul_qk);
      g.add_op(scale_div);
      g.add_op(mask_add);
      g.add_op(softmax);
      g.add_op(matmul_v);
      g.add_op(transpose);
      g.add_op(reshape);
      g.finalize();

      std::vector<partition> partitions = g.get_partitions();
      assert(partitions.size() == 1);
      partition sdp_partition = partitions[0];
      insert_in_partition_cache(patternID, sdp_partition);
      graph_partition_iter = partition_map_lookup(patternID);

    }

    graph_partition = graph_partition_iter->second;
    compiledPartitionEntry.partition_ = graph_partition;
    // partition compilation
    compile_partition(compiledPartitionEntry, eng);

    // partition execution
    auto& inputs = compiledPartitionEntry.inputLogicalTensors_;
    auto& outputs = compiledPartitionEntry.outputLogicalTensors_;
    compiledPartitionEntry.inputLLGATensors_.reserve(inputs.size());
    compiledPartitionEntry.outputLLGATensors_.reserve(outputs.size());
    allocate_sycl_graph_mem(
        compiledPartitionEntry.inputLLGATensors_, inputs[0], eng, query);
    allocate_sycl_graph_mem(
        compiledPartitionEntry.inputLLGATensors_, inputs[1], eng, key);
    allocate_sycl_graph_mem(
        compiledPartitionEntry.inputLLGATensors_,
        inputs[2],
        eng,
        softmax_scale1);
    allocate_sycl_graph_mem(
        compiledPartitionEntry.inputLLGATensors_, inputs[3], eng, attn_mask);
    allocate_sycl_graph_mem(
        compiledPartitionEntry.inputLLGATensors_, inputs[4], eng, value);
    allocate_sycl_graph_mem(
        compiledPartitionEntry.outputLLGATensors_, outputs[0], eng, output);
    compiledPartitionEntry.cp_.execute(
        strm,
        compiledPartitionEntry.inputLLGATensors_,
        compiledPartitionEntry.outputLLGATensors_);
    strm.wait();
    // cache the compiled kernel
    insert_in_fused_kernel_cache(map_key, compiledPartitionEntry);

  } else {
    cp_entry& cp = iter->second->second;

    // partition execution
    auto& inputs = cp.inputLogicalTensors_;
    auto& outputs = cp.outputLogicalTensors_;
    cp.inputLLGATensors_.reserve(inputs.size());
    cp.outputLLGATensors_.reserve(outputs.size());
    allocate_sycl_graph_mem(cp.inputLLGATensors_, inputs[0], eng, query);
    allocate_sycl_graph_mem(cp.inputLLGATensors_, inputs[1], eng, key);
    allocate_sycl_graph_mem(
        cp.inputLLGATensors_, inputs[2], eng, softmax_scale1);
    allocate_sycl_graph_mem(cp.inputLLGATensors_, inputs[3], eng, attn_mask);
    allocate_sycl_graph_mem(cp.inputLLGATensors_, inputs[4], eng, value);
    allocate_sycl_graph_mem(cp.outputLLGATensors_, outputs[0], eng, output);
    cp.cp_.execute(strm, cp.inputLLGATensors_, cp.outputLLGATensors_);
    strm.wait();
  }
}

inline Tensor _scaled_dot_product_onednn_graph_dnnl_impl(
   const Tensor& _query,
   const Tensor& _key,
   const Tensor& _value,
   const c10::optional<Tensor>& attn_mask,
   bool is_causal,
   double dropout_p,
   c10::optional<double> scale) {

 auto output = at::empty_like(_query);
 int batch_size = _query.size(0);
 int num_head = _query.size(1);
 int size_per_head = _query.size(3);
 int seq_len_q = _query.size(2);
 int seq_len_k = _key.size(2);

 const double softmax_scale =
     scale.has_value() ? scale.value() : (1.0 / std::sqrt(_query.size(-1)));
 // need contiguous to get strided layout in broadcast case
 const Tensor attn_mask_final =
     attn_mask.has_value() ? attn_mask.value().contiguous() : at::empty_like(_query);

   gpu_float_sdpa(
       batch_size,
       seq_len_q,
       seq_len_k,
       num_head,
       size_per_head,
       _query,
       _key,
       _value,
       attn_mask_final,
       softmax_scale,
       output);

 return output;
}


inline Tensor _scaled_dot_product_onednn_graph_xetla_impl(
   const Tensor& _query,
   const Tensor& _key,
   const Tensor& _value,
   const c10::optional<Tensor>& attn_mask,
   bool is_causal,
   double dropout_p,
   c10::optional<double> scale) {

 auto output = at::empty_like(_query);
 int batch_size = _query.size(0);
 int num_head = _query.size(1);
 int size_per_head = _query.size(3);
 int seq_len_q = _query.size(2);
 int seq_len_k = _key.size(2);

 const double softmax_scale =
     scale.has_value() ? scale.value() : (1.0 / std::sqrt(_query.size(-1)));
 const Tensor attn_mask_final =
     attn_mask.has_value() ? attn_mask.value() : at::empty_like(_query);

   gpu_float_sdpa(
       batch_size,
       seq_len_q,
       seq_len_k,
       num_head,
       size_per_head,
       _query,
       _key,
       _value,
       attn_mask_final,
       softmax_scale,
       output);

 return output;
}

 inline Tensor _scaled_dot_product_efficient_attention_impl(
     const Tensor& _query,
     const Tensor& _key,
     const Tensor& _value,
     const c10::optional<Tensor>& attn_mask,
     bool is_causal,
     double dropout_p,
     c10::optional<double> scale) {
 #if defined(USE_XETLA)
   // check attn_mask padded
   uint32_t attn_mask_padded_block_size = 0;
   if (attn_mask.has_value()) {
     attn_mask_padded_block_size = attn_mask.value().size(-1);
     TORCH_CHECK(
         (attn_mask_padded_block_size * _key.itemsize() % 8 == 0),
         "XeTLA SDP Attention mask needs 8bytes aligned on leading dimension ...");
   }
   // make q, k, v strided
   auto query = _query.transpose(1, 2).contiguous().transpose(1, 2);
   auto key = _key.transpose(1, 2).contiguous().transpose(1, 2);
   auto value = _value.transpose(1, 2).contiguous().transpose(1, 2);
   // create strided output
   // size [bs, num_head, qsize, head_size]
   // layout [bs, qsize, num_head, head_size]
   auto output = at::empty_like(query);
   auto dpcpp_queue = dpcppGetCurrentQueue();

   const double softmax_scale =
       scale.has_value() ? scale.value() : (1.0 / std::sqrt(query.size(-1)));

   gpu::xetla::fmha_forward_kernel(
       dpcpp_queue,
       query.data_ptr(),
       key.data_ptr(),
       value.data_ptr(),
       /* alibi */ nullptr,
       attn_mask.has_value() ? attn_mask.value().data_ptr() : (void*)nullptr,
       /* dropout_mask */ nullptr,
       output.data_ptr(),
       softmax_scale,
       /* beta */ 1.0f,
       dropout_p,
       query.size(0),
       query.size(1),
       query.size(3),
       query.size(2),
       key.size(2),
       /* ablibi padded size */ 0,
       attn_mask_padded_block_size,
       is_causal,
       false);

   return output;
 #else
   AT_ERROR("SDP: xetla library not found in compilation");
   // TODO: sycl kernel impl for efficient_attention
   // auto result = naive_scaled_dot_product(query, key, value, is_causal);
   // return std::forward_as_tuple(std::get<0>(result), std::get<1>(result));
 #endif
}

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math_impl(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    const c10::optional<Tensor>& dropout_mask,
    c10::optional<double> scale) {
  if (query_.is_nested() || key.is_nested() || value.is_nested()) {
    TORCH_CHECK(
        query_.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
        "scaled_dot_product_attention: If inputs are nested tensors they must be contiguous");
  }
  auto attn_mask = attn_mask_;
  // Naive, composite implementation defined here.

  // [Original] Scale q, k before matmul for stability see
  // https://tinyurl.com/sudb9s96 for math
  // Here we apply scaling after matmul for op fusion purpose
  bool is_negative_scaling = scale.has_value() && scale.value() < 0.0;
  const auto orig_scaling_factor = sdp::calculate_scale(
      query_, is_negative_scaling ? std::abs(scale.value()) : scale);

  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query_.is_nested() && !key.is_nested(),
        "_scaled_dot_product_attention: Nested tensors for query / key are not supported when is_causal=True");

    // Replace attn_mask with causal mask; lower triangular elements take part
    // in attention.
    const auto L = query_.sym_size(-2), S = key.sym_size(-2);
    attn_mask =
        at::ones_symint({L, S}, query_.options().dtype(at::kBool)).tril();
    attn_mask = sdp::convert_boolean_attn_mask(attn_mask, query_.dtype());
  }

  Tensor attn;
  if (attn_mask.has_value()) {
    attn_mask = attn_mask->contiguous();
    if (is_negative_scaling) {
      attn = trans_matmul_div_add(
          key,
          /*dim1=*/-1,
          /*dim2=*/-1,
          query_,
          c10::SymFloat(0.0) - orig_scaling_factor,
          *attn_mask,
          1.0);
    } else {
      attn = trans_matmul_div_add(
          key,
          /*dim1=*/-1,
          /*dim2=*/-1,
          query_,
          orig_scaling_factor,
          *attn_mask,
          1.0);
    }
  } else {
    if (is_negative_scaling) {
      attn = trans_matmul_div_scalar(
          key,
          /*dim1=*/-1,
          /*dim2=*/-1,
          query_,
          c10::SymFloat(0.0) - orig_scaling_factor);
    } else {
      attn = trans_matmul_div_scalar(
          key, /*dim1=*/-1, /*dim2=*/-1, query_, orig_scaling_factor);
    }
  }
  attn = at::softmax(attn, -1);
  if (dropout_p > 0.0) {
    if (dropout_mask.has_value()) {
      // In order to validate the correctness of the fused kernels, we need to
      // use the same dropout mask in order to compare the results.
      TORCH_WARN_ONCE("Dropout mask should only be used for testing purposes.");
      attn = attn.masked_fill(dropout_mask->logical_not(), 0.0);
      auto dropout_scaling = 1.0 / (1 - dropout_p);
      return std::make_tuple(at::matmul(attn, value * dropout_scaling), attn);
    } else {
      attn = at::dropout(attn, dropout_p, true);
    }
  }

  return std::make_tuple(at::matmul(attn, value), attn);
}

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    const c10::optional<Tensor>& dropout_mask,
    c10::optional<double> scale) {
  // on ATSM, the efficient_attention path is not available
  // With naive math path, oneDNN matmul has overflow issue with fp16 inputs
  // as a WA, convert fp16 inputs to fp32
  return IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      query_.scalar_type(),
      "scaled_dot_product_attention_math",
      [&] {
        bool is_half = std::is_same<scalar_t, at::Half>::value;
        if (is_half) {
          Tensor query_fp32 = query_.to(at::kFloat);
          Tensor key_fp32 = key.to(at::kFloat);
          Tensor value_fp32 = value.to(at::kFloat);
          auto [attn_output, attn_weight] =
              _scaled_dot_product_attention_math_impl(
                  query_fp32,
                  key_fp32,
                  value_fp32,
                  attn_mask_,
                  dropout_p,
                  is_causal,
                  dropout_mask,
                  scale);
          return std::make_tuple(
              attn_output.to(at::kHalf), attn_weight.to(at::kHalf));
        }
        return _scaled_dot_product_attention_math_impl(
            query_,
            key,
            value,
            attn_mask_,
            dropout_p,
            is_causal,
            dropout_mask,
            scale);
      });
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
_scaled_dot_product_efficient_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
#if defined(USE_ONEDNN_GRAPH)
  #if defined(USE_XETLA)
    auto out = _scaled_dot_product_onednn_graph_xetla_impl(
        query, key, value, attn_bias, is_causal, dropout_p, scale);
  #else
    auto out = _scaled_dot_product_onednn_graph_dnnl_impl(
        query, key, value, attn_bias, is_causal, dropout_p, scale);
  #endif
#else
  auto out = _scaled_dot_product_efficient_attention_impl(
      query, key, value, attn_bias, is_causal, dropout_p, scale);
#endif
  auto softmax_lse = at::empty(
      {query.size(0), query.size(1), query.size(2)},
      query.options().dtype(at::kFloat));
  Tensor seed_t = at::empty({}, at::dtype(at::kLong).device(at::kXPU));
  Tensor offset_t = at::empty({}, at::dtype(at::kLong).device(at::kXPU));
  return std::make_tuple(
      std::move(out),
      std::move(softmax_lse),
      std::move(seed_t),
      std::move(offset_t));
}

inline bool xetla_supported(Tensor q, Tensor k, Tensor v, bool is_training) {
  bool is_supported = false;
#if defined(USE_XETLA)
  DeviceId curDevID;
  AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
  if (q.dtype() == at::kHalf && k.dtype() == at::kHalf &&
      v.dtype() == at::kHalf && !is_training &&
      Settings::I().has_2d_block_array(curDevID)) {
    if ((q.size(-1) * sizeof(at::Half) % 128 == 0) &&
        (v.size(-1) * sizeof(at::Half) % 128 == 0))
      is_supported = true;
  }
#endif
  return is_supported;
}

int64_t _fused_sdp_choice(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  bool is_training =
      (query.requires_grad() || key.requires_grad() || value.requires_grad());
  // We have implemented efficient_attention backend with xetla, flash_attention
  // backend is not supported now, which will be implemented in the future. So
  // we provide two backends here.
#if defined(USE_ONEDNN_GRAPH)
  sdp::SDPBackend backend = sdp::SDPBackend::efficient_attention;
#else
  sdp::SDPBackend backend = xetla_supported(query, key, value, is_training)
      ? sdp::SDPBackend::efficient_attention
      : sdp::SDPBackend::math;
#endif
  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the fused kernels.");
  }
  return static_cast<int64_t>(backend);
}

inline void validate_sdpa_input(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  TORCH_CHECK(
      query_.dtype() == key.dtype() && query_.dtype() == value.dtype(),
      "Expected query, key, and value to have the same dtype, but got query.dtype: ",
      query_.dtype(),
      " key.dtype: ",
      key.dtype(),
      " and value.dtype: ",
      value.dtype(),
      " instead.");
  TORCH_CHECK(
      query_.device() == key.device() && query_.device() == value.device(),
      "Expected query, key, and value to have the same device type, but got query.device: ",
      query_.device(),
      " key.device: ",
      key.device(),
      " and value.device: ",
      value.device(),
      " instead.");
  TORCH_CHECK(
      query_.dim() >= 2 && key.dim() >= 2 && value.dim() >= 2,
      "Expected query, key, and value to all be  at least 2 dimensional, but got query.dim: ",
      query_.dim(),
      " key.dim: ",
      key.dim(),
      " and value.dim: ",
      value.dim(),
      " instead.");
  if (attn_mask_.has_value()) {
    auto mask_dtype = attn_mask_->dtype();
    TORCH_CHECK(
        mask_dtype == at::kBool || mask_dtype == query_.dtype(),
        "Expected attn_mask dtype to be bool or to match query dtype, but got attn_mask.dtype: ",
        mask_dtype,
        " and  query.dtype: ",
        query_.dtype(),
        " instead.");
  }
  return;
}

Tensor xetla_fsdp_forward_atten_mask_alibi_strided(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& alibi,
    const c10::optional<Tensor>& attn_mask,
    const c10::optional<Tensor>& head_mask,
    const double alpha,
    const double beta,
    const double dropout_p,
    bool is_causal,
    bool seq_last) {
  TORCH_CHECK(
      !head_mask.has_value(),
      "Unsupported feature in fsdp kernel, head_mask ...");

  TORCH_CHECK(
      query.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      key.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      value.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");

  auto output = at::empty_like(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();
  RECORD_FUNCTION("xetla_fsdp_forward_atten_mask_alibi_strided", {});

  // check alibi padded
  uint32_t alibi_padded_block_size = 0;
  if (alibi.has_value()) {
    alibi_padded_block_size = alibi.value().size(-1);
    TORCH_CHECK(
        (alibi_padded_block_size * key.itemsize() % 8 == 0),
        "XeTLA SDP Alibi needs 8bytes aligned on leading dimension ...");
  }

  // check attn_mask padded
  uint32_t attn_mask_padded_block_size = 0;
  if (attn_mask.has_value()) {
    attn_mask_padded_block_size = attn_mask.value().size(-1);
    TORCH_CHECK(
        (attn_mask_padded_block_size * key.itemsize() % 8 == 0),
        "XeTLA SDP Attention mask needs 8bytes aligned on leading dimension ...");
  }

#if defined(USE_XETLA)
  gpu::xetla::fmha_forward_kernel(
      dpcpp_queue,
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      alibi.has_value() ? alibi.value().data_ptr() : (void*)nullptr,
      attn_mask.has_value() ? attn_mask.value().data_ptr() : (void*)nullptr,
      nullptr,
      output.data_ptr(),
      alpha,
      beta,
      dropout_p,
      query.size(0),
      query.size(1),
      query.size(3),
      query.size(2),
      key.size(2),
      alibi_padded_block_size,
      attn_mask_padded_block_size,
      is_causal,
      seq_last);
#else
  AT_ERROR("SDP: xetla library not found in compilation");
#endif
  return output;
}

// @brief
// *query       shape  : [bs * beam, num_head, q_seq_len, head_dim]
//              layout : [q_seq_len, bs * beam, num_head, head_dim]
// *key         shape  : [bs, num_head, kv_in_len, head_dim]
//              layout : [kv_in_len, bs, num_head, head_dim]
// *value       shape  : [bs, num_head, kv_in_len, head_dim]
//              layout : [kv_in_len, bs, num_head, head_dim]
// *key_cache   shape  : [bs * beam, num_head, kv_out_len, head_dim]
//              layout : [kv_out_len, bs * beam, num_head, head_dim]
// *value_cache shape  : [bs * beam, num_head, kv_out_len, head_dim]
//              layout : [kv_out_len, bs * beam, num_head, head_dim]
// *index       shape  : [kv_out_len, bs * beam]
//              layout : [kv_out_len, bs * beam]
// *output      shape  : [bs * beam, num_head, kv_in_len + kv_out_len, head_dim]
//              layout : [bs * beam, kv_in_len + kv_out_len, num_head, head_dim]
// *timestep           : current time step of output seq
Tensor xetla_fsdp_index_forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& key_cache,
    const Tensor& value_cache,
    const Tensor& index,
    const c10::optional<Tensor>& alibi,
    const c10::optional<Tensor>& attn_mask,
    const c10::optional<Tensor>& head_mask,
    const int64_t timestep,
    const double alpha,
    const double beta,
    const double dropout_p,
    bool is_causal) {
  TORCH_CHECK(
      !head_mask.has_value(),
      "Unsupported feature in fsdp kernel, head_mask ...");

  TORCH_CHECK(
      query.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      key.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      value.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");

  // check alibi padded
  uint32_t alibi_padding = 0;
  if (alibi.has_value()) {
    alibi_padding = alibi.value().size(-1);
    TORCH_CHECK(
        (alibi_padding * key.itemsize() % 8 == 0),
        "XeTLA SDP Alibi needs 8bytes aligned on leading dimension ...");
  }

  // check attn_mask padded
  uint32_t attn_mask_padding = 0;
  if (attn_mask.has_value()) {
    attn_mask_padding = attn_mask.value().size(-1);
    TORCH_CHECK(
        (attn_mask_padding * key.itemsize() % 8 == 0),
        "XeTLA SDP Attention mask needs 8bytes aligned on leading dimension ...");
  }

  uint32_t beam_width = query.size(0) / key.size(0);
  TORCH_CHECK(
      beam_width == 1 || beam_width == 4,
      "SDP only support greedy search and beam search with beam size is 1 or 4");
  uint32_t num_keys_in = key.size(2);
  uint32_t num_keys_out = key_cache.size(2);
  auto output = at::empty_like(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();
  RECORD_FUNCTION("xetla_fsdp_index_forward", {});

#if defined(USE_XETLA)
  gpu::xetla::fmha_forward_index_kernel(
      dpcpp_queue,
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      key_cache.data_ptr(),
      value_cache.data_ptr(),
      index.data_ptr<int32_t>(),
      alibi.has_value() ? alibi.value().data_ptr() : (void*)nullptr,
      attn_mask.has_value() ? attn_mask.value().data_ptr() : (void*)nullptr,
      nullptr, /* dropout */
      output.data_ptr(),
      timestep,
      alpha,
      beta,
      dropout_p,
      key.size(0),
      beam_width,
      query.size(1),
      query.size(3),
      query.size(2),
      num_keys_in,
      num_keys_out,
      alibi_padding,
      attn_mask_padding,
      is_causal);
#else
  AT_ERROR("SDP: xetla library not found in compilation");
#endif
  return output;
}
} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "xetla_fsdp_forward_atten_mask_alibi_strided.xpu",
      at::AtenIpexTypeXPU::xetla_fsdp_forward_atten_mask_alibi_strided,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "xetla_fsdp_index_forward.xpu",
      at::AtenIpexTypeXPU::xetla_fsdp_index_forward,
      c10::DispatchKey::XPU);
}
} // namespace
