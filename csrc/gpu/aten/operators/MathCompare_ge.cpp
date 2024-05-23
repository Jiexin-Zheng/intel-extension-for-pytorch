#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/ScalarOps.h>
#include <ATen/quantized/QTensorImpl.h>

#include <ATen/native/TensorIterator.h>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
struct GeKernelDpcppFunctor {
  bool operator()(scalar_t a, scalar_t b) const {
    return Numerics<scalar_t>::ge(a, b);
  }
};

void ge_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "ge_dpcpp",
      [&]() {
        GeKernelDpcppFunctor<scalar_t> f;
        dpcpp_kernel_with_scalars(iter, f);
      });
}

} // namespace impl

/*=========================== ge ==========================*/

Tensor& ge_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  impl::ge_kernel_dpcpp(iter);
  return out;
}

Tensor ge(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::ge_out(self, other, result);
}

Tensor& ge_out(const Tensor& self, const Scalar& other_, Tensor& out) {
  at::AtenIpexTypeXPU::ge_out(self, wrapped_scalar_tensor(other_), out);
  return out;
}

Tensor ge(const Tensor& self, const Scalar& other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::ge_out(
      self, wrapped_scalar_tensor(other_), result);
}

} // namespace AtenIpexTypeXPU
} // namespace at
