import itertools
import logging
import numbers
import threading
from functools import lru_cache
from math import prod
from typing import Callable, Dict, List, Sequence

import torch
from torch._inductor.decomposition import select_decomp_table
from torch.fx import Graph, GraphModule, Node, subgraph_rewriter
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torch.fx.passes.utils.matcher_utils import InternalMatch
from torch.utils._pytree import tree_flatten, tree_map
from torch._dynamo.utils import detect_fake_mode

import intel_extension_for_pytorch._C._onednn_graph as onednn_graph
from torch._inductor.lowering import lowerings as torch_lowerings
from torch._inductor.lowering import register_lowering as torch_register_lowering
from torch._inductor.lowering import fallbacks, add_needs_realized_inputs, add_layout_constraint
from torch._inductor.decomposition import decompositions, get_decompositions
from torch._inductor.ir import TensorBox
from ..onednn_graph_ir import OneDNNGraphKernel

aten = torch.ops.aten
prims = torch.ops.prims
oneDNN_Graph_fusion_ops = {}

log = logging.getLogger(__name__)


class oneDNNGraphManager:
    def __init__(self, device_type, device_id):
        self.graph = onednn_graph.graph(device_type)
        self.engine = onednn_graph.engine(device_type, device_id)
        self.stream = onednn_graph.stream(self.engine)
        self.desc_id_to_node_map: dict[int, Node] = {}
        self.desc_to_scalar_data: dict[int, numbers.Number] = {}
        self.frozen_input_descs = set()
        self.dtype_map: dict[torch.dtype, onednn_graph.logical_tensor] = {
            torch.float16: onednn_graph.logical_tensor.f16,
            torch.bfloat16: onednn_graph.logical_tensor.bf16,
            torch.float32: onednn_graph.logical_tensor.f32,
            torch.int32: onednn_graph.logical_tensor.s32,
            torch.int8: onednn_graph.logical_tensor.s8,
            torch.uint8: onednn_graph.logical_tensor.u8,
            torch.bool: onednn_graph.logical_tensor.boolean,
            torch.int64: onednn_graph.logical_tensor.dt_undef,
            torch.float64: onednn_graph.logical_tensor.dt_undef,
            torch.complex64: onednn_graph.logical_tensor.dt_undef,
        }
        self.reverse_dtype_map: dict[onednn_graph.logical_tensor, torch.dtype] = {
            self.dtype_map[key]: key for key in self.dtype_map
        }
        self.current_id = 0
        self.desc_id_to_queried_desc = {}
        self.desc_ids_with_any_layout = set()
        self.partitions = None
        self.is_inference = False

    def get_partitions(self, policy=onednn_graph.partition.fusion):
        if self.partitions:
            return self.partitions
        self.partitions = self.graph.get_partitions(policy)
        self.set_any_layout()
        return self.partitions

    # based on
    # https://github.com/oneapi-src/oneDNN/blob/6dbeffbae1f23cbbeae17adb7b5b13f1f37c080e/tests/benchdnn/graph/helpers_any_layout.hpp#L29-L96
    # Set tensor layout to "any" when used only by onednn_graph partitions
    def set_any_layout(self, partitions=None):
        if partitions is None:
            partitions = self.partitions
        self.desc_ids_with_any_layout = set()
        # map from output id to all supported flags of supported partitions
        output_to_flag_map = {}

        # Initialize map of supported tensors
        for p in self.partitions:
            p_is_supported = p in partitions and p.is_supported()
            for out_desc in p.get_output_ports():
                id = out_desc.get_id()
                if p_is_supported and id not in output_to_flag_map:
                    output_to_flag_map[id] = []
            for in_desc in p.get_input_ports():
                id = in_desc.get_id()
                if id in output_to_flag_map:
                    output_to_flag_map[id].append(p_is_supported)
        for p in partitions:
            if not p.is_supported():
                continue
            for in_desc in p.get_input_ports():
                id = in_desc.get_id()
                if id not in output_to_flag_map:
                    continue
                flags = output_to_flag_map[id]
                # if all uses of in_desc are supported use "any"
                if all(flags):
                    self.desc_ids_with_any_layout.add(id)

    def update_input_descs(
        self,
        descs: List[onednn_graph.logical_tensor],
        aten_inputs: List[torch.Tensor],
        cache_weight: bool = False,
    ):
        frozen_input_descs_set = self.get_frozen_input_descs()
        assert len(descs) == len(aten_inputs)
        for i, (d, at_in) in enumerate(zip(descs, aten_inputs)):
            make_constant = cache_weight and isinstance(at_in, torch.nn.Parameter)
            make_constant = make_constant or (cache_weight and d.get_id() in frozen_input_descs_set)
            property_type = (
                onednn_graph.logical_tensor.property_type.constant
                if make_constant
                else onednn_graph.logical_tensor.property_type.variable
            )
            descs[i] = onednn_graph.logical_tensor(
                d.get_id(),
                d.get_data_type(),
                at_in.size(),
                at_in.stride(),
                property_type,
            )
        return descs

    def get_compiled_output_descs(
        self, cp: onednn_graph.compiled_partition, out_descs: List[onednn_graph.logical_tensor]
    ):
        return [cp.query_logical_tensor(desc.get_id()) for desc in out_descs]

    def compile_partition(self, p: onednn_graph.partition, inputs: List[onednn_graph.logical_tensor]):
        outputs = p.get_output_ports()
        for i, input in enumerate(inputs):
            id = input.get_id()
            if id in self.desc_id_to_queried_desc:
                inputs[i] = self.desc_id_to_queried_desc[id]
        for i, output in enumerate(outputs):
            id = output.get_id()
            if id in self.desc_ids_with_any_layout:
                outputs[i] = onednn_graph.logical_tensor(
                    id,
                    output.get_data_type(),
                    output.get_dims(),
                    onednn_graph.logical_tensor.layout_type.any,
                    output.get_property_type(),
                )
        cpart = p.compile(inputs, outputs, self.engine)
        for i, output in enumerate(outputs):
            id = output.get_id()
            outputs[i] = cpart.query_logical_tensor(id)
            self.desc_id_to_queried_desc[id] = outputs[i]
        return cpart

    def add_op(self, op_kind, name, inputs, outputs=None, kwargs=None):
        if outputs is None:
            outputs = []
        if outputs:
            id = outputs[0].get_id()
        else:
            id = self.generate_id()
        op = onednn_graph.op(id, op_kind, name)
        if kwargs is None:
            kwargs = {}
        for attr_key in kwargs:
            if isinstance(attr_key, str):
                if hasattr(onednn_graph.op, attr_key):
                    op.set_attr(getattr(onednn_graph.op, attr_key), kwargs[attr_key])
            else:
                op.set_attr(attr_key, kwargs[attr_key])
        if op_kind == onednn_graph.op.Concat:
            inputs = inputs[0]
        op.add_inputs(inputs)
        if outputs:
            op.add_outputs(outputs)
        self.graph.add_op(op, False)
        return id

    def generate_id(self):
        id = self.current_id
        self.current_id += 1
        return id

    def create_onednn_descs_from_node(
        self, node: Node, ptype=onednn_graph.logical_tensor.property_type.variable
    ) -> List[onednn_graph.logical_tensor]:
        assert node.op in ["placeholder", "call_function"]
        if isinstance(node.meta["val"], (list, tuple)):
            outputs = [
                self.create_onednn_desc_from_meta(val, ptype)
                for val in node.meta["val"]
            ]
        else:
            outputs = [self.create_onednn_desc_from_meta(node.meta["val"], ptype)]
        for desc in outputs:
            self.register_node_by_desc(node, desc)
        return outputs

    def create_onednn_desc_from_meta(
        self, val, ptype=onednn_graph.logical_tensor.property_type.variable
    ) -> onednn_graph.logical_tensor:
        if isinstance(val, torch.SymInt):
            dtype = onednn_graph.logical_tensor.dt_undef
            size = [1]
            stride = [1]
        else:
            dtype = self.dtype_map[val.dtype]
            size = list(val.size())
            stride = list(val.stride())
        # TODO: workaround to reset stride due to oneDNN bug
        # for cases stride is larger than number of elements in tensor
        # and in decreasing order (contiguous)
        if (
            len(size)
            and size[0] == 1
            and stride[0] > prod(size)
            and stride == sorted(stride, reverse=True)
        ):
            stride[0] = prod(size)
        onednn_desc = onednn_graph.logical_tensor(
            self.generate_id(), dtype, size, stride, ptype
        )
        return onednn_desc

    def create_onednn_desc_from_scalar(self, scalar, dtype=None) -> onednn_graph.logical_tensor:
        if not isinstance(scalar, torch.Tensor):
            fake_tensor = torch.tensor([scalar], dtype=dtype)
            desc = self.create_onednn_desc_from_meta(
                fake_tensor, ptype=onednn_graph.logical_tensor.property_type.constant
            )
            self.register_scalar_data(scalar, desc, dtype)
        else:
            desc = self.create_onednn_desc_from_meta(
                scalar, ptype=onednn_graph.logical_tensor.property_type.constant
            )
            self.register_frozen_input_descs(desc)
            self.register_scalar_data(scalar.data, desc, dtype)
        return desc

    def overwrite_scalar_args(self, args, cast_scalar=True):
        args = list(args)
        if cast_scalar:
            assert len(args) == 2
            assert any(isinstance(arg, onednn_graph.logical_tensor) for arg in args)
            for arg in args:
                if isinstance(arg, onednn_graph.logical_tensor):
                    torch_type = self.reverse_dtype_map[arg.get_data_type()]
                    break
        for arg_idx, arg in enumerate(args):
            # We only handle scalars, not lists of constant scalars
            if isinstance(arg, numbers.Number):
                # arg is a scalar value, we get a logical_tensor of shape=()
                cast_type = torch_type if cast_scalar else None
                args[arg_idx] = self.create_onednn_desc_from_scalar(arg, cast_type)
        return args

    def register_node_by_desc(self, node, desc):
        if isinstance(desc, onednn_graph.logical_tensor):
            desc = desc.get_id()
        self.desc_id_to_node_map[desc] = node

    def get_node_from_desc(self, desc):
        if isinstance(desc, onednn_graph.logical_tensor):
            desc = desc.get_id()
        return self.desc_id_to_node_map[desc]

    def register_scalar_data(self, data, desc, dtype=None):
        if isinstance(desc, onednn_graph.logical_tensor):
            desc = desc.get_id()
        self.desc_to_scalar_data[desc] = (data, dtype)

    def get_scalar_data_from_desc(self, desc):
        if isinstance(desc, onednn_graph.logical_tensor):
            desc = desc.get_id()
        return self.desc_to_scalar_data[desc]

    def register_frozen_input_descs(self, desc):
        if isinstance(desc, onednn_graph.logical_tensor):
            desc = desc.get_id()
        self.frozen_input_descs.add(desc)

    def get_frozen_input_descs(self):
        return self.frozen_input_descs

    def get_args_to_onednn_partition_order(self, onednn_partition, node_args):
        onednn_input_names = [
            in_desc.get_id()
            if in_desc.get_id() in self.desc_to_scalar_data.keys()
            else self.get_node_from_desc(in_desc).name
            for in_desc in onednn_partition.get_input_ports()
        ]
        arg_names = [n.name for n in node_args]

        # Get the index in args if arg exists in args, otherwise get the relevant scalar data from graph
        args_to_onednn_order = []
        for name in onednn_input_names:
            if name in arg_names:
                args_to_onednn_order.append(arg_names.index(name))
            else:
                scalar, dtype = self.get_scalar_data_from_desc(name)
                if not isinstance(scalar, torch.Tensor):
                    scalar = torch.tensor([scalar], dtype=dtype)
                if hasattr(scalar, "constant"):
                    scalar = scalar.constant
                args_to_onednn_order.append(scalar)
        return args_to_onednn_order

def onednn_graph_fuse_fx(gm_graph: GraphModule, is_inference: bool=True):
    log.debug("Compiling graph with oneDNN backend")
    gm = gm_graph.owning_module
    log.debug(gm.print_readable(print_output=False))
    rewrite_graph(gm)
    log.debug("Build oneDNN graph")
    onednn_graph = build_onednn_graph(gm)
    onednn_graph.is_inference = is_inference
    log.debug("Fuse fx graph based on oneDNN graph partitions")
    fuse_graph(gm, onednn_graph)
    log.debug("Re-apply Inductor Decomps after fusion for any un-lowered ops")
    reapply_decomps(gm)
    log.debug("Finished compiling graph with oneDNN backend")
    log.debug("====== Fx Graph after oneDNN compile ======")
    log.debug(gm.print_readable(print_output=False))
    return gm

onednn_graph_intensive_ops = [
    aten.addmm.default,
    aten.avg_pool2d.default,
    aten.bmm.default,
    aten.batch_norm.default,
    aten.baddbmm.default,
    aten.convolution.default,
    aten.layer_norm.default,
    aten.max_pool2d.default,
    aten.max_pool3d.default,
    aten.mm.default,
    #aten.native_batch_norm.default,
    aten.native_layer_norm.default,
    aten.cat.default,
]


def get_filtered_partitions(
    onednn_graph_manager: oneDNNGraphManager, allowed_ops=onednn_graph_intensive_ops
):
    """
    Get onednn Graph partitions which include at least one op which is in allowed_ops.
    By default this function will return partitions which are intensive,
    i.e. defined in onednn_graph_intensive_ops and known to have performant onednn code.

    Args:
        ``onednn_graph_manager``: The oneDNNGraphManager containing lowered ops to partition
        ``allowed_ops``: The list of aten.ops which will allow a partition to be lowered

    Returns:
        ``filtered_partitions``: List[onednn_graph.partition]: A list of partitions
        which contain at least one intensive op
        ``filtered_node_lists``: List[List[Node]]: A list of Nodes in the
        fx Graph which correspond to each partition
    """

    onednn_graph_partitions = onednn_graph_manager.get_partitions()
    supported_partitions = [lp for lp in onednn_graph_partitions if lp.is_supported()]
    node_lists = [
        [onednn_graph_manager.get_node_from_desc(id) for id in lp.get_ops()]
        for lp in supported_partitions
    ]

    if not allowed_ops:
        return supported_partitions, node_lists

    filtered_partitions = []
    filtered_node_lists = []
    for partition, nodes in zip(supported_partitions, node_lists):
        if any(node.target in allowed_ops for node in nodes):
            filtered_partitions.append(partition)
            filtered_node_lists.append(nodes)
    # Need to reset descs with any_layout since set of supported partitions is different
    onednn_graph_manager.set_any_layout(filtered_partitions)
    return filtered_partitions, filtered_node_lists


@lru_cache(maxsize=None)
def allocate_empty_aten_from_desc(desc: onednn_graph.logical_tensor) -> torch.Tensor:
    cpu_tensor = torch.empty(desc.get_dims())
    return cpu_tensor.to("xpu")

def fallback_handler(kernel, add_to_fallback_set=True):
    if add_to_fallback_set:
        fallbacks.add(kernel)

    def handler(*args, **kwargs):
        return tree_map(
            TensorBox.create, OneDNNGraphKernel.create(kernel, *args, **kwargs)
        )

    return handler

# register oneDNNGraph kernel for inductor lowering
def register_oneDNNGraph_kernel(op, layout_constraint=None):
    assert op not in decompositions, f"both a fallback and a decomp for same op: {op}"

    def register_fallback(op_overload):
        add_needs_realized_inputs(op_overload)
        if layout_constraint is not None:
            add_layout_constraint(op_overload, layout_constraint)
        return torch_register_lowering(op_overload, type_promotion_kind=None)(
            fallback_handler(op_overload)
        )

    if isinstance(op, OnednnGraphPartitionModule):
        register_fallback(op)
    else:
        raise RuntimeError(f"Unsupported fallback {op} with type {type(op)}")


class OnednnGraphPartitionModule:
    def __init__(
        self,
        onednn_graph_manager: oneDNNGraphManager,
        partition: onednn_graph.partition,
        input_order_data: List = None,
        name="",
    ):
        super().__init__()
        self.__name__ = name
        self.partition = partition
        self.onednn_graph_manager = onednn_graph_manager
        self.input_order_data = [] if input_order_data is None else input_order_data
        self.input_descs = partition.get_input_ports()
        self.kernel = None
        self.output_descs = None

        # assume static shape
        # cache the onednn graph tensors
        self.input_onednn_tensors = []
        self.output_onednn_tensors = []
        # cache the output tensors to avoid reallocation
        self.output_tensors = []

        self.lock = threading.Lock()

    def name(self):
        return self.__name__

    def __call__(self, *args):
        if self not in torch_lowerings:
            register_oneDNNGraph_kernel(self)
            return torch_lowerings[self](*args)
        # If val is an int, then it gives the index of args, otherwise it is a scalar tensor so we use as-is
        input_tensors = [
            args[val] if isinstance(val, int) else val for val in self.input_order_data
        ]

        # TODO: remove detect_fake_mode with an meta impl
        fake_mode = detect_fake_mode(args)
        if fake_mode:
            input_descs = self.onednn_graph_manager.update_input_descs(
                self.input_descs, input_tensors
            )
            compiled_partition = self.onednn_graph_manager.compile_partition(
                self.partition, input_descs
            )
            output_descs = self.onednn_graph_manager.get_compiled_output_descs(
                compiled_partition, self.partition.get_output_ports()
            )
            output_tensors = [
                allocate_empty_aten_from_desc(out_desc)
                for out_desc in output_descs
            ]
            return output_tensors[0] if len(output_tensors) == 1 else output_tensors

        with self.lock:
            if not self.kernel:
                cache_parameter = self.onednn_graph_manager.is_inference
                self.input_descs = self.onednn_graph_manager.update_input_descs(
                    self.input_descs, input_tensors, cache_parameter
                )
                self.kernel = self.onednn_graph_manager.compile_partition(
                    self.partition, self.input_descs
                )
                self.output_descs = self.onednn_graph_manager.get_compiled_output_descs(
                    self.kernel, self.partition.get_output_ports()
                )

        if not self.input_onednn_tensors:
            self.input_onednn_tensors = [
                onednn_graph.tensor(
                    input_desc, self.onednn_graph_manager.engine, input_tensor.data_ptr()
                )
                for input_desc, input_tensor in zip(self.input_descs, input_tensors)
            ]
        else:
            for onednn_t, aten_t in zip(self.input_onednn_tensors, input_tensors):
                onednn_t.set_data_handle(aten_t.data_ptr())

        if not self.output_tensors:
            self.output_tensors = [
                allocate_empty_aten_from_desc(out_desc)
                for out_desc in self.output_descs
            ]

        if not self.output_onednn_tensors:
            self.output_onednn_tensors = [
                onednn_graph.tensor(
                    output_desc, self.onednn_graph_manager.engine, self.output_tensor.data_ptr()
                )
                for output_desc, self.output_tensor in zip(
                    self.output_descs, self.output_tensors
                )
            ]

        assert not any(
            isinstance(out, torch._subclasses.FakeTensor) for out in self.output_tensors
        ), "unexpected faketensor in output_tensors"

        #with record_function(f"onednn_fuse_{self.__name__}"):
        self.kernel.execute(
            self.onednn_graph_manager.stream,
            self.input_onednn_tensors,
            self.output_onednn_tensors,
        )
        # TODO: It seems like this fix and also the return statements of def call_function should be handled differently.
        return (
            self.output_tensors[0]
            if len(self.output_tensors) == 1
            else self.output_tensors
        )

def build_onednn_graph(gm: GraphModule) -> oneDNNGraphManager:
    onednn_graph_manager = oneDNNGraphManager(onednn_graph.engine.gpu, 0)

    graph_input_nodes = list(filter(lambda n: n.op == "placeholder", gm.graph.nodes))

    class FusionInterpreter(torch.fx.Interpreter):
        def run_node(self, node):
            self.current_node = node
            return super().run_node(node)

        def placeholder(self, target, args, kwargs):
            # Add an input to graph
            return onednn_graph_manager.create_onednn_descs_from_node(self.current_node)[0]

        def call_function(self, target, args, kwargs):
            # Jiexin: To convert frozen parameters: Some of them will be torch.Tensor instances here
            args = [onednn_graph_manager.create_onednn_desc_from_scalar(arg, 
                        arg.dtype) 
                    if isinstance(arg, torch.Tensor) else arg 
                    for arg in args
            ]

            # With placeholder defined, args is always a tuple of logical tensors except for the case of scalars
            if not isinstance(target, torch._ops.OpOverload):
                res = target(*args)
                if "getitem" in self.current_node.name and isinstance(
                    res, onednn_graph.logical_tensor
                ):
                    out_descs = onednn_graph_manager.create_onednn_descs_from_node(
                        self.current_node
                    )
                    onednn_graph_manager.add_op(
                        onednn_graph.op.Wildcard, self.current_node.name, [res], out_descs
                    )
                    return out_descs[0]
                return res
            out_descs = onednn_graph_manager.create_onednn_descs_from_node(self.current_node)
            if target not in lowerings or not self._is_valid_lowering(
                target, args, out_descs
            ):
                onednn_graph_manager.add_op(
                    onednn_graph.op.Wildcard,
                    self.current_node.name,
                    [
                        arg
                        for arg in tree_flatten(args)[0]
                        if isinstance(arg, onednn_graph.logical_tensor)
                    ],
                    out_descs,
                )
            else:
                # target in lowerings, add node to onednn_graph
                lowerings[target](
                    onednn_graph_manager, self.current_node.name, args, out_descs, dict(kwargs)
                )
            if isinstance(self.current_node.meta["val"], Sequence):
                return out_descs
            return out_descs[0]

        def output(self, target, args, kwargs):
            flat_out, _ = tree_flatten(args[0])
            for i, output_logten in enumerate(flat_out):
                if isinstance(
                    output_logten, onednn_graph.logical_tensor
                ) and output_logten.get_data_type() in [
                    onednn_graph.logical_tensor.f32,
                    onednn_graph.logical_tensor.f16,
                    onednn_graph.logical_tensor.bf16,
                ]:
                    endop_id = onednn_graph_manager.add_op(
                        onednn_graph.op.End,
                        self.current_node.name + str(output_logten.get_id()),
                        [output_logten],
                        [],
                    )
                    onednn_graph_manager.register_node_by_desc(self.current_node, endop_id)

        def _is_valid_lowering(self, target, args, out_descs):
            valid_types = [
                onednn_graph.logical_tensor.f32,
                onednn_graph.logical_tensor.f16,
                onednn_graph.logical_tensor.bf16,
            ]

            if (
                (target == aten.layer_norm and len(args[1]) > 1)
                or (target == aten.batch_norm and None in args[:5])
                or any(
                    arg.get_data_type() not in valid_types
                    for arg in args
                    if isinstance(arg, onednn_graph.logical_tensor)
                )
                or out_descs[0].get_data_type() not in valid_types
            ):
                return False
            return True

    onednn_graph_args = tuple(graph_input_nodes)
    FusionInterpreter(gm).run(*onednn_graph_args)
    onednn_graph_manager.graph.finalize()
    return onednn_graph_manager


def fuse_graph(gm: GraphModule, onednn_graph_manager: oneDNNGraphManager) -> GraphModule:
    supported_partitions, node_lists = get_filtered_partitions(onednn_graph_manager)

    if len(node_lists) == 0:
        return

    gm = fuse_by_partitions(gm, node_lists)

    for node in gm.graph.nodes:
        if node.op == "call_module" and "fused_" in node.name:
            partition_idx = int(node.name.split("fused_")[1])
            current_partition = supported_partitions[partition_idx]
            current_partition_nodes = node_lists[partition_idx]
            current_partition_ext_name = "_".join(
                [n.name for n in current_partition_nodes]
            )

            out_descs = current_partition.get_output_ports()
            if len(out_descs) == 1:
                onednn_graph_manager.register_node_by_desc(node, out_descs[0])
            else:
                getitem_users = [
                    node for node in node.users.keys() if "getitem" in node.name
                ]
                for getitem in getitem_users:
                    ind = getitem.args[1]
                    onednn_graph_manager.register_node_by_desc(getitem, out_descs[ind])

            args_to_onednn_order = onednn_graph_manager.get_args_to_onednn_partition_order(
                current_partition, node.args
            )

            gm.delete_submodule(node.target)
            node.name = f"onednn_{node.name}"
            node.op = "call_function"          
            node.target = OnednnGraphPartitionModule(
                onednn_graph_manager,
                current_partition,
                input_order_data=args_to_onednn_order,
                name=current_partition_ext_name,
            )
            oneDNN_Graph_fusion_ops[current_partition_ext_name] = node.target
            setattr(node.target, "_inductor_lowering_function", True)
            log.info("Using oneDNN fusion: %s", current_partition_ext_name)

    gm.recompile()
    return gm


def replace_pattern_with_replacement(
    gm: GraphModule,
    pattern: Callable,
    replacement: Callable,
    filters: List[Callable[[InternalMatch, Graph, Graph], bool]] = None,
):
    matches_and_replacements = subgraph_rewriter.replace_pattern_with_filters(
        gm, pattern, replacement, filters
    )
    # We need to reassign the meta (FakeTensor) from old nodes to the new
    for match in matches_and_replacements:
        if match.replacements:
            old_meta = match.nodes_map[match.anchor].meta
            nodes = [
                node
                for node in match.replacements
                if isinstance(node.target, torch._ops.OpOverload)
            ]
            nodes[0].meta = old_meta
    return matches_and_replacements


def replace_t_matmul_to_matmul(gm: GraphModule):
    def matmul_patterns_generator(func, transpose_a, transpose_b):
        # Need to check single/double transpose cases separately since number of args differs
        def single_addmm_pattern(bias, mat1, mat2, inds1):
            return func(
                bias,
                aten.permute.default(mat1, inds1) if transpose_a else mat1,
                aten.permute.default(mat2, inds1) if transpose_b else mat2,
            )

        def single_addmm_replacement(bias, mat1, mat2, inds1):
            return func(
                bias, mat1, mat2, transpose_a=transpose_a, transpose_b=transpose_b
            )

        def double_addmm_pattern(bias, mat1, mat2, inds1, inds2):
            return func(
                bias,
                aten.permute.default(mat1, inds1) if transpose_a else mat1,
                aten.permute.default(mat2, inds2) if transpose_b else mat2,
            )

        def double_addmm_replacement(bias, mat1, mat2, inds1, inds2):
            return func(
                bias, mat1, mat2, transpose_a=transpose_a, transpose_b=transpose_b
            )

        def single_mm_pattern(mat1, mat2, inds1):
            return func(
                aten.permute.default(mat1, inds1) if transpose_a else mat1,
                aten.permute.default(mat2, inds1) if transpose_b else mat2,
            )

        def single_mm_replacement(mat1, mat2, inds1):
            return func(mat1, mat2, transpose_a=transpose_a, transpose_b=transpose_b)

        def double_mm_pattern(mat1, mat2, inds1, inds2):
            return func(
                aten.permute.default(mat1, inds1) if transpose_a else mat1,
                aten.permute.default(mat2, inds2) if transpose_b else mat2,
            )

        def double_mm_replacement(mat1, mat2, inds1, inds2):
            return func(mat1, mat2, transpose_a=transpose_a, transpose_b=transpose_b)

        if func == aten.addmm.default:
            if transpose_a and transpose_b:
                return double_addmm_pattern, double_addmm_replacement
            else:
                return single_addmm_pattern, single_addmm_replacement
        else:
            if transpose_a and transpose_b:
                return double_mm_pattern, double_mm_replacement
            else:
                return single_mm_pattern, single_mm_replacement

    def ind_filter(match: InternalMatch, graph1: Graph, graph2: Graph):
        # If transpose inds are the last two dims, replacement is valid
        transpose_nodes = [
            match.nodes_map[n]
            for n in match.nodes_map
            if hasattr(n, "target") and n.target == aten.permute.default
        ]
        for node in transpose_nodes:
            ndims = len(node.meta["val"].shape)
            inds = node.args[1]
            for i, ind in enumerate(inds[:-2]):
                if i != ind:
                    return False
            # Handle negative inds using %
            min_ind = min(node.args[1][-2:]) % ndims
            max_ind = max(node.args[1][-2:]) % ndims
            if min_ind != ndims - 2 or max_ind != min_ind + 1:
                return False
        return True

    matches = []
    for func, t1, t2 in itertools.product(
        [
            aten.addmm.default,
            aten.mm.default,
            aten.bmm.default,
        ],
        [True, False],
        [True, False],  # transpose a, b
    ):
        if not t1 and not t2:
            continue
        pat, rep = matmul_patterns_generator(func, t1, t2)
        matches += replace_pattern_with_replacement(gm, pat, rep, [ind_filter])

    return matches


def allow_manydim_bmm(gm: GraphModule):
    # Pytorch bmm only allows 3 dims, but MatMul in oneDNN Graph allows
    # an arbitrary number of dims, so we can simplify the graph to match
    # existing semi-compiler patterns
    def pattern_generator(view_func1, view_func2, view_func3):
        def bmm_pattern(input1, shape1, input2, shape2, shape3):
            x1 = view_func1(input1, shape1)
            x2 = view_func2(input2, shape2)
            x3 = aten.bmm.default(x1, x2)
            return view_func3(x3, shape3)

        return bmm_pattern

    def bmm_replacement(input1, shape1, input2, shape2, shape3):
        return aten.bmm.default(input1, input2)

    def view_filter(match: InternalMatch, graph1: Graph, graph2: Graph):
        # Replacement is valid if view ops are all not changing last two dims, so only batch changes.
        after_view_node = match.nodes_map[match.anchors[0]]
        before_view_node0 = after_view_node.args[0].args[0]
        before_view_node1 = after_view_node.args[0].args[1]
        valid_match = True
        for node in [after_view_node, before_view_node0, before_view_node1]:
            valid_match &= node.args[0].meta["val"].shape[-2:] == torch.Size(
                node.args[1][-2:]
            )
        if(valid_match == False):
            return valid_match
        # To make it run into a 3-dim bmm kernel in oneDNN when the original bmm
        # input n-dim is 5 and its first dim is 1, since this kind of case 
        # always runs into a ref kernel on GPU. (tmp fix for hf_Reformer and 
        # hf_BigBird, etc.)
        # [1, 12, 64, 64, 128] --> [768, 64, 128]
        not_redundant_five_dims = True
        for node in [after_view_node, before_view_node0, before_view_node1]:
            not_redundant_five_dims &= not ( (node.args[0].meta["val"].shape[0] == 1) and
                (len(node.args[0].meta["val"].shape) == 5))
        valid_match = not_redundant_five_dims and valid_match
        return valid_match

    matches = []
    view_ops = [
        aten.view.default,
        aten.reshape.default,
    ]
    for func1 in view_ops:
        for func2 in view_ops:
            for func3 in view_ops:
                matches += replace_pattern_with_replacement(
                    gm,
                    pattern_generator(func1, func2, func3),
                    bmm_replacement,
                    [view_filter],
                )
    return matches


def _redundant_filter(match: InternalMatch, graph1: Graph, graph2: Graph):
    # Return true if match is "redundant", ie output has same shape and stride as input
    node = match.nodes_map[match.anchors[0]]
    return (
        node.args[0].meta["val"].shape == node.meta["val"].shape
        and node.args[0].meta["val"].stride() == node.meta["val"].stride()
    )


def remove_redundant_expand(gm: GraphModule):
    def expand_pattern(input, shape):
        return aten.expand.default(input, shape)

    def expand_replacement(input, shape):
        return input

    return replace_pattern_with_replacement(
        gm, expand_pattern, expand_replacement, [_redundant_filter]
    )


def replace_max_pool_with_indices(
    gm: torch.fx.GraphModule, num_dims=2
) -> torch.fx.GraphModule:
    assert num_dims in [2, 3]
    default_args = [None, None, None, [0] * num_dims, [1] * num_dims, False]
    pattern_func = (
        aten.max_pool2d_with_indices.default
        if num_dims == 2
        else aten.max_pool3d_with_indices.default
    )
    replace_func = aten.max_pool2d.default if num_dims == 2 else aten.max_pool3d.default

    def pattern3(self, kernel_size, stride):
        return pattern_func(self, kernel_size, stride)[0]

    def pattern4(self, kernel_size, stride, padding):
        return pattern_func(self, kernel_size, stride, padding)[0]

    def pattern5(self, kernel_size, stride, padding, dilation):
        return pattern_func(self, kernel_size, stride, padding, dilation)[0]

    def pattern6(self, kernel_size, stride, padding, dilation, ceil_mode):
        return pattern_func(self, kernel_size, stride, padding, dilation, ceil_mode)[0]

    def replacement3(self, kernel_size, stride):
        return replace_func(
            self, kernel_size, stride, default_args[3], default_args[4], default_args[5]
        )

    def replacement4(self, kernel_size, stride, padding):
        return replace_func(
            self, kernel_size, stride, padding, default_args[4], default_args[5]
        )

    def replacement5(self, kernel_size, stride, padding, dilation):
        return replace_func(
            self, kernel_size, stride, padding, dilation, default_args[5]
        )

    def replacement6(self, kernel_size, stride, padding, dilation, ceil_mode):
        return replace_func(self, kernel_size, stride, padding, dilation, ceil_mode)

    matches = []
    matches += replace_pattern_with_replacement(gm, pattern3, replacement3)
    matches += replace_pattern_with_replacement(gm, pattern4, replacement4)
    matches += replace_pattern_with_replacement(gm, pattern5, replacement5)
    matches += replace_pattern_with_replacement(gm, pattern6, replacement6)
    return matches


def rewrite_graph(gm):
    # Remove expand nodes that have identical input and output shape
    remove_redundant_expand(gm)
    # Replace view(a) + view(b) + bmm(a,b) + view() with a N-dim call to MatMul
    allow_manydim_bmm(gm)
    # Replace (transpose + (add/b)mm) with single call to addmm with kwarg "transpose_=True" to integrate with oneDNN API
    replace_t_matmul_to_matmul(gm)
    # Replace aten::max_pool{2,3}d_with_indices to aten::max_pool{2,3}d because
    # MaxPool currently doesn't support with indices
    replace_max_pool_with_indices(gm, num_dims=2)
    replace_max_pool_with_indices(gm, num_dims=3)


def replace_node_with_subgraph(node: Node, subgraph: Graph):
    val_map: Dict[Node, Node] = {}
    node_inputs = [arg for arg in node.args if isinstance(arg, Node)]
    replacement_placeholders = [n for n in subgraph.nodes if n.op == "placeholder"]
    for ni, rp in zip(node_inputs, replacement_placeholders):
        val_map[rp] = ni
    [
        subgraph.erase_node(n)
        for n in reversed(subgraph.nodes)
        if len(n.users) == 0 and n.op != "output"
    ]
    with node.graph.inserting_before(node):
        copied_returning_node = node.graph.graph_copy(subgraph, val_map)
    if not isinstance(copied_returning_node, Node):
        copied_returning_node = copied_returning_node[0]
    node.replace_all_uses_with(copied_returning_node)
    node.graph.erase_node(node)


def reapply_decomps(gm: GraphModule):
    all_decomps = select_decomp_table()
    search_decomps = {
        key: all_decomps[key] for key in (lowerings.keys() & all_decomps.keys())
    }
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in search_decomps:
            args = [
                arg.meta["val"] if hasattr(arg, "meta") and "val" in arg.meta else arg
                for arg in node.args
            ]
            replacement = make_fx(node.target, all_decomps)(*args)
            replace_node_with_subgraph(node, replacement.graph)
    gm.graph.eliminate_dead_code()
    gm.recompile()


# define lowerings for aten IR to oneDNN IR
lowerings = {}


def register_lowering(aten_op_packet):
    def decorator(fn):
        def _register_lowering(aten_op):
            if isinstance(aten_op, torch._ops.OpOverloadPacket):
                overload_funcs = [
                    getattr(aten_op, name) for name in aten_op.overloads()
                ]
            elif isinstance(aten_op, torch._ops.OpOverload):
                overload_funcs = [aten_op]
            else:
                raise RuntimeError(f"Attempting to lower unsupported op {aten_op}")
            for overload_fn in overload_funcs:
                if overload_fn in lowerings:
                    raise RuntimeError(
                        f"Already registered for {overload_fn.name()} in lowerings"
                    )
                lowerings[overload_fn] = fn

        tree_map(_register_lowering, aten_op_packet)
        return fn

    return decorator


class LoweringConfig:
    def __init__(
        self,
        onednn_graph_op,
        in_descs_filtering=None,
        attribute_inputs=None,
        scalar_in_descs=False,
    ):
        """
        Map aten op to onednn_graph op for lowering.

        Args:
            ``onednn_graph_op``: The onednn_graph Op for lowering
            ``in_descs_filtering``: A lambda function which is used to reorder or filter
            the input logical tensors `in_descs`. For example,
            `lambda in_descs: [in_descs[1], in_descs[2], in_descs[0]]` could be used to
            reorder the inputs. For convenience, the string "first" can be used in place of
            `lambda in_descs: in_descs[:1]`
            ``attribute_inputs``: A dictionary with onednn_graph op attributes as keys, and
            lambda functions to retrieve the value from the input logical tensors `in_descs`
            ``scalar_in_descs``: bool, True if scalar inputs should be cast to match other input types
        """
        self.onednn_graph_op = onednn_graph_op
        self.in_descs_filtering_func = in_descs_filtering
        if in_descs_filtering == "first":
            self.in_descs_filtering_func = lambda in_descs: in_descs[:1]
        self.attribute_inputs = {} if attribute_inputs is None else attribute_inputs
        self.scalar_in_descs = scalar_in_descs


_lowerings_map = {
    aten.add: LoweringConfig(onednn_graph.op.Add, scalar_in_descs=True),
    aten.addmm: LoweringConfig(
        onednn_graph.op.MatMul, lambda in_descs: [in_descs[1], in_descs[2], in_descs[0]]
    ),
    aten.avg_pool2d: LoweringConfig(
        onednn_graph.op.AvgPool,
        "first",
        attribute_inputs={
            onednn_graph.op.strides: lambda in_descs: in_descs[2]
            if len(in_descs) >= 3
            else in_descs[1],
            onednn_graph.op.pads_begin: lambda in_descs: in_descs[3]
            if len(in_descs) >= 4
            else [0, 0],
            onednn_graph.op.pads_end: lambda in_descs: in_descs[3]
            if len(in_descs) >= 4
            else [0, 0],
            onednn_graph.op.rounding_type: lambda in_descs: (
                "ceil" if len(in_descs) == 6 and in_descs[5] else "floor"
            ),
            onednn_graph.op.exclude_pad: False,
            onednn_graph.op.kernel: lambda in_descs: in_descs[1],
            onednn_graph.op.data_format: "NCX",
        },
    ),
    aten.bmm: LoweringConfig(onednn_graph.op.MatMul),
    aten.clone: LoweringConfig(onednn_graph.op.Reorder),
    aten.div: LoweringConfig(onednn_graph.op.Divide, scalar_in_descs=True),
    aten.elu: LoweringConfig(
        onednn_graph.op.Elu,
        "first",
        attribute_inputs={onednn_graph.op.alpha: lambda in_descs: in_descs[1]},
    ),
    aten.gelu: LoweringConfig(onednn_graph.op.GELU),
    aten.hardsigmoid: LoweringConfig(
        onednn_graph.op.HardSigmoid, attribute_inputs={onednn_graph.op.alpha: 1 / 6, onednn_graph.op.beta: 0.5}
    ),
    aten.hardswish: LoweringConfig(onednn_graph.op.HardSwish),
    aten.hardtanh: LoweringConfig(
        onednn_graph.op.Clamp,
        "first",
        attribute_inputs={
            onednn_graph.op.min: lambda in_descs: float(in_descs[1]),
            onednn_graph.op.max: lambda in_descs: float(in_descs[2]),
        },
    ),
    aten.leaky_relu: LoweringConfig(
        onednn_graph.op.LeakyReLU,
        "first",
        attribute_inputs={
            onednn_graph.op.alpha: lambda in_descs: 0.01 if len(in_descs) == 1 else in_descs[1]
        },
    ),
    aten.mish: LoweringConfig(onednn_graph.op.Mish),
    aten.max_pool2d: LoweringConfig(
        onednn_graph.op.MaxPool,
        "first",
        attribute_inputs={
            onednn_graph.op.strides: lambda in_descs: in_descs[2],
            onednn_graph.op.pads_begin: lambda in_descs: in_descs[3]
            if len(in_descs) >= 4
            else [0, 0],
            onednn_graph.op.pads_end: lambda in_descs: in_descs[3]
            if len(in_descs) >= 4
            else [0, 0],
            onednn_graph.op.kernel: lambda in_descs: in_descs[1],
            onednn_graph.op.rounding_type: lambda in_descs: (
                "ceil" if len(in_descs) == 6 and in_descs[5] else "floor"
            ),
            onednn_graph.op.dilations: lambda in_descs: in_descs[4]
            if len(in_descs) >= 5
            else [1, 1],
            onednn_graph.op.data_format: "NCX",
        },
    ),
    aten.mm: LoweringConfig(onednn_graph.op.MatMul),
    aten.mul: LoweringConfig(onednn_graph.op.Multiply, scalar_in_descs=True),
    # # Jiexin: Only mapped BatchNormInference currently, need to add trainning 
    # # case in future
    # aten.native_batch_norm: LoweringConfig(onednn_graph.op.BatchNormInference, 
    #     lambda in_descs: in_descs[0:6],
    #     attribute_inputs={onednn_graph.op.epsilon: lambda in_descs: in_descs[-1]}),
    aten.native_layer_norm: LoweringConfig(onednn_graph.op.LayerNorm, 
        lambda in_descs: [in_descs[0], in_descs[2], in_descs[3]],
        attribute_inputs={onednn_graph.op.epsilon: lambda in_descs: in_descs[-1],
            onednn_graph.op.begin_norm_axis: lambda in_descs: in_descs[1]},),
    aten.cat: LoweringConfig(
        onednn_graph.op.Concat,
        lambda in_descs: in_descs[:-1],
        attribute_inputs={onednn_graph.op.axis: lambda in_descs: in_descs[-1]},
    ),
    aten.permute: LoweringConfig(
        onednn_graph.op.StaticTranspose,
        "first",
        attribute_inputs={onednn_graph.op.order: lambda in_descs: in_descs[1]},
    ),
    aten.relu: LoweringConfig(onednn_graph.op.ReLU),
    aten.sigmoid: LoweringConfig(onednn_graph.op.Sigmoid),
    aten.sub: LoweringConfig(onednn_graph.op.Subtract, scalar_in_descs=True),
    aten.tanh: LoweringConfig(onednn_graph.op.Tanh),
    aten._softmax: LoweringConfig(
        onednn_graph.op.SoftMax,
        "first",
        attribute_inputs={onednn_graph.op.axis: lambda in_descs: in_descs[1]},
    ),
}

lowering_functions = []
for op in _lowerings_map:

    @register_lowering(op)
    def _lowering(
        onednn_graph_manager,
        node_name,
        in_descs,
        out_descs,
        kwargs=None,
        lowering_config=_lowerings_map[op],
    ):
        if kwargs is None:
            kwargs = {}
        for attr_kind in lowering_config.attribute_inputs:
            attr_value = lowering_config.attribute_inputs[attr_kind]
            if callable(attr_value):
                # Lambda functions used to get attribute value from in_descs
                kwargs[attr_kind] = attr_value(in_descs)
            else:
                kwargs[attr_kind] = attr_value
        if lowering_config.in_descs_filtering_func:
            # Lambda function used to filter or reorder in_descs
            in_descs = lowering_config.in_descs_filtering_func(in_descs)
        if lowering_config.scalar_in_descs:
            # If scalar inputs should be cast to match other input type
            in_descs = onednn_graph_manager.overwrite_scalar_args(in_descs)
        return onednn_graph_manager.add_op(
            lowering_config.onednn_graph_op, node_name, in_descs, out_descs, kwargs
        )

    lowering_functions.append(_lowering)


@register_lowering(aten.convolution)
def _onednn_convolution(onednn_graph_manager, node_name, in_descs, out_descs, kwargs):
    # in_descs[2] is None when optional bias is False
    input_len = 2 if len(in_descs) > 2 and in_descs[2] is None else 3
    if in_descs[6]:  # if ConvTranspose
        return onednn_graph_manager.add_op(
            onednn_graph.op.ConvTranspose,
            node_name,
            in_descs[:input_len],
            out_descs,
            {
                onednn_graph.op.strides: in_descs[3],
                onednn_graph.op.pads_begin: in_descs[4],
                onednn_graph.op.pads_end: in_descs[4],
                onednn_graph.op.dilations: in_descs[5],
                onednn_graph.op.output_padding: in_descs[7],
                onednn_graph.op.groups: in_descs[8],
                onednn_graph.op.data_format: "NCX",
                onednn_graph.op.weights_format: "IOX",
            },
        )
    else:
        return onednn_graph_manager.add_op(
            onednn_graph.op.Convolution,
            node_name,
            in_descs[:input_len],
            out_descs,
            {
                onednn_graph.op.strides: in_descs[3],
                onednn_graph.op.pads_begin: in_descs[4],
                onednn_graph.op.pads_end: in_descs[4],
                onednn_graph.op.dilations: in_descs[5],
                onednn_graph.op.groups: in_descs[8],
                onednn_graph.op.data_format: "NCX",
                onednn_graph.op.weights_format: "OIX",
            },
        )