import torch
from .overrides import override_size_asserts
from torch._inductor.codegen.common import register_backend_for_device
from .codegen.triton import XPUTritonScheduling
from .codegen.wrapper import XPUTritonWrapperCodeGen

from .lowering import *
from .fx_passes.fusion import *
from ._meta_registrations import *
from torch._inductor import config

if torch.xpu.is_available():
    override_size_asserts()
    register_backend_for_device("xpu", XPUTritonScheduling, XPUTritonWrapperCodeGen)
    if config.onednn_graph:
        # enable oneDNN Graph fusion pass
        from .fx_passes.onednn_graph_fusion import onednn_graph_fuse_fx
        config.post_grad_custom_pre_pass = onednn_graph_fuse_fx
