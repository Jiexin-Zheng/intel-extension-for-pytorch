import torch
from typing import List, Optional, Any, Set
from torch._inductor.ir import (
    ExternKernelAlloc,
    MutationLayout,
    Layout,
    FixedLayout,
    FlexibleLayout,
    NoneLayout,
    convert_shape_to_inductor,
    ir_node_to_tensor,
    may_convert_to_optional,
    mark_node_as_mutating,
    TensorBox,
)
from torch._prims_common import (
    make_channels_last_strides_for,
)
from torch._inductor.virtualized import V
from torch._inductor.utils import pad_listlike

import sympy


def _prepare_convolution_fusion_create(
    cls,
    x: "TensorBox",
    weight: "TensorBox",
    bias: "TensorBox",
    padding: List[int],
    stride: List[int],
    dilation: List[int],
    groups: int,
    transposed: bool = False,
    output_padding: List[int] = None,
):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for convolution post-op fusion's create function, including deciding the output
    layout (channels first or channels last), realizing inputs and make them etc. The
    function only supports the XPU device since conv post-op fusion kernel is only
    supported on XPU right now.
    """

    # Port from aten/src/ATen/native/ConvUtils.h: _conv_input_size
    def _conv_input_size(
        output_size, weight_size, padding, output_padding, stride, dilation, groups
    ):
        assert len(output_size) == len(weight_size), "Expect input dim == weight dim"
        dim = len(output_size)
        assert dim > 2, "Expect input dim > 2"

        BATCH_DIM = 0
        WEIGHT_INPUT_CHANNELS_DIM = 1
        input_size = []
        input_size.append(output_size[BATCH_DIM])
        input_size.append(weight_size[WEIGHT_INPUT_CHANNELS_DIM] * groups)
        for d in range(2, dim):
            kernel = (weight_size[d] - 1) * dilation[d - 2] + 1
            input_size_d = (
                (output_size[d] - 1) * stride[d - 2]
                - (padding[d - 2] * 2)
                + kernel
                + output_padding[d - 2]
            )
            input_size.append(input_size_d)
        return list(map(int, input_size))

    x.realize()
    weight.realize()
    if bias is not None:
        bias.realize()
    with V.graph.fake_mode:
        x_fake = ir_node_to_tensor(x, guard_shape=True)
        weight_fake = ir_node_to_tensor(weight, guard_shape=True)
        dims = len(x_fake.size()) - 2
        assert 0 < len(padding) <= dims
        assert 0 < len(dilation) <= dims
        assert 0 < len(stride) <= dims
        padding = pad_listlike(padding, dims)
        dilation = pad_listlike(dilation, dims)
        stride = pad_listlike(stride, dims)
        if output_padding is None:
            output_padding = pad_listlike([0], dims)
        else:
            assert 0 < len(output_padding) <= dims
            output_padding = pad_listlike(output_padding, dims)

        assert isinstance(groups, int)
        bias_fake = (
            ir_node_to_tensor(bias, guard_shape=True) if bias is not None else bias
        )
        output = torch.ops.aten.convolution(
            x_fake,
            weight_fake,
            bias_fake,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
        output_size = output.size()

        req_stride_order = [0] + list(reversed(range(1, len(stride) + 1)))
        req_stride_order = [len(req_stride_order)] + req_stride_order
        output_stride = make_channels_last_strides_for(output_size)

    x = cls.require_stride_order(x, req_stride_order)
    inputs = [x, weight]

    kernel_layout = FixedLayout(
        x.get_device(),
        x.get_dtype(),
        convert_shape_to_inductor(output_size),
        convert_shape_to_inductor(output_stride),
    )
    constant_args = [padding, stride, dilation, groups]

    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    return inputs, constant_args, kernel_layout, req_stride_order


class ConvolutionUnary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.torch_ipex._convolution_pointwise",
            cpp_kernel_name="torch_ipex::_convolution_pointwise",
        )
        self.cpp_kernel_key = "convolution_pointwise"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.get_kernel_name(),
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        attr,
        scalars: Optional[List[Any]],
        algorithm,
    ):
        (inputs, constant_args, kernel_layout, _) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        constant_args = constant_args + [
            attr,
            may_convert_to_optional(scalars),
            algorithm,
        ]
        return ConvolutionUnary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )


class LinearUnary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.torch_ipex._linear_pointwise",
            cpp_kernel_name="torch_ipex::_linear_pointwise",
        )
        self.cpp_kernel_key = "linear_pointwise"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                c10::string_view attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.get_kernel_name(),
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
        )

    @classmethod
    def create(cls, x, w, b, attr, scalars, algorithm):
        x = cls.require_contiguous(cls.realize_input(x))
        w = cls.require_contiguous(cls.realize_input(w))

        *m, ic = x.get_size()
        # After freezing pass, w should be [ic, oc] shape, format is ba
        ic, oc = w.get_size()
        inputs = [x, w]
        constant_args = [attr, scalars if scalars else [-1], algorithm]
        if b is not None:
            b = cls.require_contiguous(cls.realize_input(b))
            inputs.append(b)
        else:
            constant_args.insert(0, None)

        return LinearUnary(
            layout=FlexibleLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=list(m) + [oc],
            ),
            inputs=inputs,
            constant_args=constant_args,
        )

    def apply_constraint(self):
        pass


class LinearBinary(ExternKernelAlloc):
    kernel = "torch.ops.torch_ipex._linear_pointwise.binary"

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.torch_ipex._linear_pointwise.binary",
            cpp_kernel_name="torch_ipex::_linear_pointwise",
        )
        self.cpp_kernel_overload_name = "binary"
        self.cpp_kernel_key = "linear_pointwise_binary"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& other_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                c10::string_view attr)
        """

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.get_kernel_name(),
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )

    @classmethod
    def create(cls, x, y, w, b, attr):
        x = cls.require_contiguous(cls.realize_input(x))
        y = cls.require_contiguous(cls.realize_input(y))
        w = cls.require_contiguous(cls.realize_input(w))

        *m, ic = x.get_size()
        # After freezing pass, w should be [ic, oc] shape, format is ba
        ic, oc = w.get_size()

        inputs = [x, y, w]
        constant_args = [attr]
        if b is not None:
            b = cls.require_contiguous(cls.realize_input(b))
            inputs.append(b)
        else:
            constant_args.insert(0, b)

        return LinearBinary(
            layout=FlexibleLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=list(m) + [oc],
            ),
            inputs=inputs,
            constant_args=constant_args,
        )

    def apply_constraint(self):
        pass


class ConvolutionBinary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        cpp_constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.torch_ipex._convolution_pointwise.binary",
            cpp_kernel_name="torch_ipex::_convolution_pointwise",
        )
        self.cpp_kernel_overload_name = "binary"
        self.cpp_kernel_key = "convolution_pointwise_binary"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                at::Tensor& other_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view binary_attr,
                c10::optional<at::Scalar> alpha,
                c10::optional<c10::string_view> unary_attr,
                torch::List<c10::optional<at::Scalar>> unary_scalars,
                c10::optional<c10::string_view> unary_algorithm)"""
        self.cpp_constant_args = cpp_constant_args

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.get_kernel_name(),
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        other: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        binary_attr: str,
        binary_alpha: Optional[float],
        unary_attr: Optional[str],
        unary_scalars: Optional[List[Any]],
        unary_algorithm: Optional[str],
    ):
        (
            inputs,
            constant_args,
            kernel_layout,
            req_stride_order,
        ) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        other = cls.require_stride_order(other, req_stride_order)
        inputs.insert(1, other)
        constant_args = constant_args + [
            binary_attr,
            binary_alpha,
            unary_attr,
            may_convert_to_optional(unary_scalars),
            unary_algorithm,
        ]
        return ConvolutionBinary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )


class ConvolutionBinaryInplace(ExternKernelAlloc):
    def __init__(
        self,
        kernel_layout,
        inputs,
        constant_args=(),
    ):
        # Due to constrain of op.call, other (Tensor&) should be at input[0]
        reordered_inputs = [inputs[1], inputs[0]] + inputs[2:]

        super().__init__(
            kernel_layout,
            reordered_inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.torch_ipex._convolution_pointwise_.binary",
            cpp_kernel_name="torch_ipex::_convolution_pointwise_",
        )
        self.cpp_kernel_overload_name = "binary"
        self.cpp_kernel_key = "convolution_pointwise_binary_"
        # TODO: op.call: input[0] should be at::Tensor&
        self.cpp_op_schema = """
            at::Tensor&(
                at::Tensor& other_t,
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view binary_attr,
                c10::optional<at::Scalar> alpha,
                c10::optional<c10::string_view> unary_attr,
                torch::List<c10::optional<at::Scalar>> unary_scalars,
                c10::optional<c10::string_view> unary_algorithm)"""

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.get_kernel_name(),
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )

    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        other: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        binary_attr: str,
        binary_alpha: Optional[float],
        unary_attr: Optional[str],
        unary_scalars: Optional[List[Any]],
        unary_algorithm: Optional[str],
    ):
        (
            inputs,
            constant_args,
            _,
            req_stride_order,
        ) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        other = cls.require_stride_order(other, req_stride_order)
        inputs.insert(1, other)
        constant_args = constant_args + [
            binary_attr,
            binary_alpha,
            unary_attr,
            may_convert_to_optional(unary_scalars),
            unary_algorithm,
        ]
        packed = ConvolutionBinaryInplace(
            kernel_layout=NoneLayout(inputs[1].get_device()),  # type: ignore[arg-type]
            inputs=inputs,
            constant_args=constant_args,
        )
        mark_node_as_mutating(packed, inputs[1])
        # This op mutates in place which means that the result is not the
        # target but rather the input that is being mutated
        # init reorders the inputs, so inputs[1] becomes packed.inputs[0]
        return packed.inputs[0]
