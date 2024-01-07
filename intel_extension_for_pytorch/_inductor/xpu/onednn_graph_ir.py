import torch
from torch._inductor.ir import ExternKernelAlloc, FixedLayout, MultiOutput, MultiOutputLayout, Layout
from typing import (
    Any,
    Sequence,
)
from torch._inductor.virtualized import V
import torch._logging
from torch._inductor.utils import (
    convert_shape_to_inductor,
)
import intel_extension_for_pytorch as ipex

aten = torch.ops.aten

class OneDNNGraphKernel(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs=None,
        schema=None,
    ):
        super().__init__(
            layout,
            tuple(tensor_args),
            tuple(nontensor_args),
        )
        # We need output buffers for generating kernel arguments in the
        # abi-compatible mode, where we retrieve outputs by pass each individual
        # output through the abi-compatible interface.
        self.outputs: Sequence[Any] = []
        self.use_runtime_dispatch = False
        self.abi_compatible_kernel = None

        assert isinstance(
            kernel,
            ipex._inductor.xpu.fx_passes.onednn_graph_fusion.OnednnGraphPartitionModule
        ), f"Fails to create OneDNNGraphKernel for {kernel}: {type(kernel)} not supported"
        self.kernel = kernel

        self.unflatten_args = unflatten_args
        self.kwargs = {} if kwargs is None else kwargs
        V.graph.warn_fallback(self.kernel)
    
    def codegen(self, wrapper):
        self.codegen_comment(wrapper)
        args = [*self.codegen_args(), *self.codegen_kwargs()]
        V.graph.wrapper_code.generate_oneDNN_Graph_kernel_alloc(self.get_name(), self.kernel.name(), args)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(cls, kernel, *args, **kwargs):
        context = V.graph.fake_mode
        with context:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                schema,
            ) = cls.process_kernel(kernel, *args, **kwargs)

        device = cls.find_device(tensor_args, example_output)
        assert device, "Not sure where to find device info"

        packed = cls(
            MultiOutputLayout(device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            schema=schema,
        )

        def generate_output(output, indices):
            if isinstance(output, (list, tuple)):
                return type(output)(
                    generate_output(output[i], indices + [(type(output), i)])
                    for i in range(len(output))
                )
            elif isinstance(output, torch.Tensor):
                return MultiOutput(
                    FixedLayout(
                        output.device,
                        output.dtype,
                        convert_shape_to_inductor(output.size()),
                        convert_shape_to_inductor(output.stride()),
                    ),
                    packed,
                    indices,
                )
            elif isinstance(output, int):
                return output
            else:
                assert (
                    output is None
                ), f"FallbackKernel output type {type(output)} is not supported"
                return None

        outputs = generate_output(example_output, [])
        return outputs

    @staticmethod
    def find_device(tensor_args, example_output):
        if tensor_args:
            return tensor_args[0].get_device()
        if isinstance(example_output, torch.Tensor):
            return example_output.device
        if isinstance(example_output, (list, tuple)):
            devices = {OneDNNGraphKernel.find_device(None, x) for x in example_output}
            # Remove None
            devices = [device for device in devices if device]
            if len(devices) == 1:
                return devices[0]
            for device in devices:
                if device.type == "cuda":
                    return device
            return devices[0]
        return None

    @staticmethod
    def tensor_to_layout(output: torch.Tensor):
        return FixedLayout(
            output.device,
            output.dtype,
            convert_shape_to_inductor(output.size()),
            convert_shape_to_inductor(output.stride()),
        )