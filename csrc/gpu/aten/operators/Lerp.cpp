#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include "ATen/OpMathType.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
bool is_lerp_weight_small(scalar_t weight) {
  return std::abs(weight) < scalar_t(0.5);
}

template <typename scalar_t>
bool is_lerp_weight_small(c10::complex<scalar_t> weight) {
  // Avoid the sqrt in abs(weight)
  return (weight.real() * weight.real() + weight.imag() * weight.imag()) <
      scalar_t(0.25);
}
template <typename scalar_t, typename opmath_t>
struct lerp_tensor_kernel_functor {
  scalar_t operator()(scalar_t self_val, scalar_t end_val, scalar_t weight_val)
      const {
    opmath_t self_val_f = self_val;
    opmath_t end_val_f = end_val;
    opmath_t weight_val_f = weight_val;
    return (is_lerp_weight_small(weight_val_f))
        ? self_val_f + weight_val_f * (end_val_f - self_val_f)
        : end_val_f - (end_val_f - self_val_f) * (opmath_t(1) - weight_val_f);
  }
};

void lerp_tensor_kernel(at::TensorIteratorBase& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "lerp_tensor_kernel",
      [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        lerp_tensor_kernel_functor<scalar_t, opmath_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
}

template <typename scalar_t, typename opmath_t>
struct LerpScalarKernelFunctor {
  scalar_t operator()(scalar_t self_val, scalar_t end_val) const {
    opmath_t self_val_f = self_val;
    opmath_t end_val_f = end_val;
    return (is_lerp_weight_small(weight_val))
        ? self_val_f + weight_val * (end_val_f - self_val_f)
        : end_val_f - (end_val_f - self_val_f) * (opmath_t(1) - weight_val);
  }

  LerpScalarKernelFunctor(opmath_t weight_val_) : weight_val(weight_val_) {}

 private:
  opmath_t weight_val;
};

void lerp_scalar_kernel(
    at::TensorIteratorBase& iter,
    const c10::Scalar& weight) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "lerp_tensor_kernel",
      [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        auto weight_val = weight.to<opmath_t>();
        LerpScalarKernelFunctor<scalar_t, opmath_t> f(weight_val);
        dpcpp_kernel_with_scalars(iter, f);
      });
}

} // namespace impl

Tensor& lerp_out(
    const Tensor& self,
    const Tensor& end,
    const Tensor& weight,
    Tensor& out) {
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(out)
                  .add_input(self)
                  .add_input(end)
                  .add_input(weight)
                  .build();
  impl::lerp_tensor_kernel(iter);
  return out;
}

Tensor& lerp_out(
    const Tensor& self,
    const Tensor& end,
    const Scalar& weight,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, end);
  impl::lerp_scalar_kernel(iter, weight);
  return out;
}

Tensor& lerp_(Tensor& self, const Tensor& end, const Tensor& weight) {
  auto iter = TensorIteratorConfig()
                  .add_output(self)
                  .add_input(self)
                  .add_input(end)
                  .add_input(weight)
                  .build();
  impl::lerp_tensor_kernel(iter);
  return self;
}

Tensor& lerp_(Tensor& self, const Tensor& end, const Scalar& weight) {
  auto iter = TensorIterator::binary_op(self, self, end);
  impl::lerp_scalar_kernel(iter, weight);
  return self;
}

Tensor lerp(const Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor result = at::empty_like(self);
  return at::AtenIpexTypeXPU::lerp_out(self, end, weight, result);
}

Tensor lerp(const Tensor& self, const Tensor& end, const Scalar& weight) {
  Tensor result = at::empty_like(self);
  return at::AtenIpexTypeXPU::lerp_out(self, end, weight, result);
}

} // namespace AtenIpexTypeXPU
} // namespace at
