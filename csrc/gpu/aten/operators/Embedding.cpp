#include <ATen/ATen.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"

#include <core/Device.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include "EmbeddingBackwardKernel.h"
#include "Indexing.h"
#include "PSTLFunctions.h"
#include "comm/ATDispatch.h"
#include "comm/Atomics.h"
#include "comm/SYCLGroupAlgorithm.h"

using namespace torch_ipex::xpu::dpcpp;
using namespace torch_ipex::xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename IdxType>
struct IndicesCountKernelFunctor {
  void operator()(sycl::item<1> item) const {
    auto row = indices[item.get_id(0)];
    atomicAdd((dpcpp_global_ptr_pt<IdxType>)(&indices_cnt[row]), 1);
  }
  IndicesCountKernelFunctor(IdxType* indices_cnt_, IdxType* indices_)
      : indices_cnt(indices_cnt_), indices(indices_) {}

 private:
  IdxType* indices_cnt;
  IdxType* indices;
};

template <typename IdxType>
static inline void indices_count(
    IdxType* indices_cnt,
    IdxType* indices,
    int64_t indices_num) {
  auto& queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(__cgh) {
    IndicesCountKernelFunctor<IdxType> kfn(indices_cnt, indices);
    __cgh.parallel_for<decltype(kfn)>(sycl::range<1>(indices_num), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename ValType, typename IdxType>
class EmbBwdOperator {
 public:
  EmbBwdOperator(IdxType* indices_cnt, IdxType padding_idx)
      : indices_cnt_(indices_cnt), padding_idx_(padding_idx) {}

  void operator()(
      ValType* dst,
      ValType* src,
      int64_t dst_off,
      int64_t src_off,
      int64_t idx,
      ValType alpha) const {
    if (padding_idx_ == idx) {
      dst[dst_off] = 0;
      return;
    }

    if (indices_cnt_ != nullptr) {
      atomicAdd(
          (dpcpp_global_ptr_pt<ValType>)(dst + dst_off),
          src[src_off] / indices_cnt_[idx]);
    } else {
      atomicAdd((dpcpp_global_ptr_pt<ValType>)(dst + dst_off), src[src_off]);
    }
  }

  IdxType* indices_cnt_;
  int64_t padding_idx_;
};

template <typename scalar_t, typename index_t>
static inline void embedding_dense_backward_kernel(
    const Tensor& grad_output,
    const Tensor& grad_weight,
    const Tensor& indices,
    const Tensor& indices_cnt,
    int64_t padding_idx) {
  TensorInfo<index_t, int64_t> indices_info =
      getTensorInfo<index_t, int64_t>(indices);
  indices_info.collapseDims();

  TensorInfo<scalar_t, int64_t> src_info =
      getTensorInfo<scalar_t, int64_t>(grad_output);

  TensorInfo<scalar_t, int64_t> dst_info =
      getTensorInfo<scalar_t, int64_t>(grad_weight);

  auto func = indices_cnt.defined()
      ? EmbBwdOperator<scalar_t, index_t>(
            indices_cnt.data_ptr<index_t>(), padding_idx)
      : EmbBwdOperator<scalar_t, index_t>(nullptr, padding_idx);

  using SrcInfo = TensorInfo<scalar_t, int64_t>;
  using DstInfo = TensorInfo<scalar_t, int64_t>;
  using IdxInfo = TensorInfo<index_t, int64_t>;
  auto cfg = IndexKernelConfig<
      SrcInfo,
      DstInfo,
      IdxInfo,
      EmbBwdOperator<scalar_t, index_t>>::
      make_config(src_info, dst_info, indices_info, 0, 0, true, func);
  launch_index_kernel(cfg);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
struct RenormKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int tid = item.get_local_linear_id();
    int sgSize = item.get_local_range(0);
    auto group_idx = item.get_group(0);
    if (group_idx >= num_unique_indices) {
      return;
    }

    int base_index = indices[group_idx] * weights_stride0;

    accscalar_t v = static_cast<accscalar_t>(0);
    for (int i = tid; i < dim; i += sgSize) {
      auto x =
          static_cast<accscalar_t>(weights[base_index + i * weights_stride1]);
      if (norm_type == 1) {
        v += std::abs(x);
      } else if (norm_type == 2) {
        v += x * x;
      } else {
        v += std::pow(x, norm_type);
      }
    }

    v = GroupReduceSumSGSizeEqualstoNumSG(
        item,
        v,
        static_cast<accscalar_t*>(
            smem.template get_multi_ptr<sycl::access::decorated::no>().get()));

    if (tid == 0) {
      smem[0] = std::pow(v, static_cast<accscalar_t>(1.0 / norm_type));
    }
    item.barrier(dpcpp_local_fence);

    if (smem[0] > max_norm) {
      auto factor = static_cast<scalar_t>(
          max_norm / (smem[0] + std::numeric_limits<accscalar_t>::epsilon()));
      for (int i = tid; i < dim; i += sgSize) {
        weights[base_index + i * weights_stride1] *= factor;
      }
    }
  }
  RenormKernelFunctor(
      scalar_t* weights_,
      index_t* indices_,
      accscalar_t max_norm_,
      accscalar_t norm_type_,
      int64_t dim_,
      int64_t weights_stride0_,
      int64_t weights_stride1_,
      int64_t num_unique_indices_,
      dpcpp_local_acc_t<accscalar_t> smem_)
      : weights(weights_),
        indices(indices_),
        max_norm(max_norm_),
        norm_type(norm_type_),
        dim(dim_),
        weights_stride0(weights_stride0_),
        weights_stride1(weights_stride1_),
        num_unique_indices(num_unique_indices_),
        smem(smem_) {}

 private:
  scalar_t* weights;
  index_t* indices;
  accscalar_t max_norm;
  accscalar_t norm_type;
  int64_t dim;
  int64_t weights_stride0;
  int64_t weights_stride1;
  int64_t num_unique_indices;
  dpcpp_local_acc_t<accscalar_t> smem;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
void renorm_kernel(
    scalar_t* weights,
    index_t* indices,
    accscalar_t max_norm,
    accscalar_t norm_type,
    int64_t dim,
    int64_t weights_stride0,
    int64_t weights_stride1,
    int64_t num_unique_indices) {
  const int64_t work_group_size = dpcppMaxWorkItemsPerEU();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto smem = dpcpp_local_acc_t<accscalar_t>(
        (work_group_size / 8) * sizeof(accscalar_t),
        cgh); // We use the smallest subgroup size to ensure enough space
    RenormKernelFunctor<scalar_t, accscalar_t, index_t> kfn(
        weights,
        indices,
        max_norm,
        norm_type,
        dim,
        weights_stride0,
        weights_stride1,
        num_unique_indices,
        smem);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(
            sycl::range<1>(work_group_size * num_unique_indices),
            sycl::range<1>(work_group_size)),
        kfn);
  };

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

} // namespace impl

template <typename index_t>
struct embedding_dense_backward_eq_functor {
  auto operator()(index_t a, index_t b) const {
    return Numerics<index_t>::eq(a, b);
  }
};

Tensor embedding_dense_backward(
    const Tensor& grad_output,
    const Tensor& indices_,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  auto grad_arg = TensorArg(grad_output, "grad", 1);
  auto indices_arg = TensorArg(indices_, "indices", 1);
  checkScalarTypes("embedding_backward", indices_arg, {kLong, kInt});
  isOnSameDevice("embedding_backward", grad_arg, indices_arg);
  auto indices = indices_.contiguous();

  auto num_indices = indices.numel();
  auto grad_output_cont =
      grad_output.contiguous().view({num_indices, grad_output.size(-1)});
  Tensor grad_weight;

  auto sorted_indices =
      at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto orig_indices = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor count;
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_output_cont.scalar_type(),
      "embedding_backward",
      [&]() {
        IPEX_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_backward", [&] {
              auto sorted_begin = sorted_indices.data_ptr<index_t>();
              auto orig_begin = orig_indices.data_ptr<index_t>();
              {
                sorted_indices.copy_(indices);
                torch_ipex::xpu::pstl::iota(
                    orig_begin, orig_begin + num_indices, (index_t)0);
                torch_ipex::xpu::pstl::sort<index_t, index_t>(
                    indices.data_ptr<index_t>(),
                    sorted_begin,
                    orig_begin,
                    num_indices,
                    false);
              }

              if (scale_grad_by_freq) {
                count = at::empty_like(sorted_indices);
                index_t* count_begin = count.data_ptr<index_t>();
                // Take the maximum of each count per unique key:
                // sorted: 2 5 5 5 7 7 8 9 9
                //  count: 1 3 3 3 2 2 1 2 2
                //
                embedding_dense_backward_eq_functor<index_t> f;
                torch_ipex::xpu::pstl::
                    count_by_segment<index_t, index_t, index_t>(
                        sorted_begin,
                        sorted_begin + num_indices,
                        count_begin,
                        f);
              }
              grad_weight = impl::
                  embedding_backward_deterministic_kernel<scalar_t, index_t>(
                      grad_output_cont,
                      orig_indices,
                      sorted_indices,
                      count,
                      num_weights,
                      padding_idx);
            });
      });
  return grad_weight;
}

struct embedding_renorm_cmp_functor {
  template <typename T>
  bool operator()(T lhs, T rhs) const {
    if (lhs != rhs) {
      return false;
    }
    return true;
  }
};

Tensor& embedding_renorm_(
    Tensor& self,
    const Tensor& indices,
    double max_norm,
    double norm_type) {
  auto self_arg = TensorArg(self, "self", 1);
  auto indices_arg = TensorArg(indices, "indices", 2);
  checkDim("embedding_renorm_", self_arg, 2);
  checkScalarTypes("embedding_renorm_", indices_arg, {kLong, kInt});

  auto indices_contig = indices.contiguous();
  auto num_indices = indices.numel();

  IPEX_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_renorm_", [&]() {
    auto num_indices = indices.numel();
    auto indices_contig = std::get<0>(indices.sort()).contiguous();
    auto unique_indices = at::empty(indices.numel(), indices.options());

    unique_indices.copy_(indices_contig);

    int64_t num_unique_indices;
    embedding_renorm_cmp_functor f;
    num_unique_indices = torch_ipex::xpu::pstl::unique<index_t, index_t>(
                             unique_indices.data_ptr<index_t>(),
                             unique_indices.data_ptr<index_t>() + num_indices,
                             f) -
        unique_indices.data_ptr<index_t>();

    int dim = self.stride(0);

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "embedding_renorm_",
        [&] {
          using accscalar_t = acc_type<scalar_t>;
          impl::renorm_kernel(
              self.data_ptr<scalar_t>(),
              unique_indices.data_ptr<index_t>(),
              static_cast<accscalar_t>(max_norm),
              static_cast<accscalar_t>(norm_type),
              dim,
              self.stride(0),
              self.stride(1),
              num_unique_indices);
        });
  });

  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
