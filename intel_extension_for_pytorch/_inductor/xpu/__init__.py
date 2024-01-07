import torch
from .overrides import override_size_asserts
from torch._inductor.codegen.common import register_backend_for_device
from .codegen.triton import XPUTritonScheduling
from .codegen.wrapper import XPUTritonWrapperCodeGen

from .lowering import *
from .fx_passes.fusion import *
from ._meta_registrations import *

if torch.xpu.is_available():
    override_size_asserts()
    register_backend_for_device("xpu", XPUTritonScheduling, XPUTritonWrapperCodeGen)
    if ipex_xpu_enable_oneDNN_graph:
        # enable oneDNN Graph fusion pass
        from torch._inductor import config
        from .fx_passes.onednn_graph_fusion import onednn_graph_fuse_fx
        config.post_grad_custom_pre_pass = onednn_graph_fuse_fx
