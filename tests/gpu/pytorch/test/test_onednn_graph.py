
import os
import sys
test_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(test_root)

import pytest
import common.xpu_test_base
# Owner(s): ["module: nn"]

import contextlib
from functools import partial
from collections import namedtuple
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.bias import CausalVariant, causal_lower_right, causal_upper_left
from torch.nn.parameter import Parameter
import unittest
from unittest.mock import patch, MagicMock, ANY
import math
import torch.optim as optim
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCUDA, onlyCPU
from typing import List, Tuple, Optional
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    TEST_WITH_ROCM,
    skipIfRocm,
    skipIfTorchDynamo,
    TEST_FAIRSEQ,
    run_tests,
    parametrize,
    freeze_rng_state,
    TEST_WITH_CROSSREF,
    slowTest,
    set_default_dtype,
    gradcheck,
    make_tensor,
    NOTEST_CPU,
    IS_WINDOWS,
    TEST_WITH_TORCHDYNAMO,
)
from torch._dynamo.testing import CompileCounterWithBackend


from torch.testing._internal.common_methods_invocations import wrapper_set_seed

if TEST_FAIRSEQ:
    import fairseq.models.transformer as fairseq_transformer

SdpaShape = namedtuple('Sdpa_Shape', ['batch', 'num_heads', 'seq_len', 'head_dim'])
Tolerances = namedtuple('Tolerances', ['atol', 'rtol'])

@contextlib.contextmanager
def use_deterministic_algorithims(mode: bool, warn_only: bool):
    r"""
    This context manager can be used to temporarily enable or disable deterministic algorithms.
    Upon exiting the context manager, the previous state of the flag will be restored.
    """
    previous_mode: bool = torch.are_deterministic_algorithms_enabled()
    previous_warn_only: bool = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(mode, warn_only=warn_only)
        yield {}
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)


# Found in torch/testing/_comparison.py
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}


def get_rtol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    deviation = torch.abs(deviation / true_value)
    # Fill in the nans with the default rtol
    torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
    return deviation.max().item()


def get_atol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    atol = torch.abs(deviation).max().item()
    return atol


def get_tolerances(
    true_value: torch.Tensor,
    computed_value: torch.Tensor,
    fudge_factor: Optional[float] = None,
) -> Tuple[float, float]:
    """Returns the absolute and relative tolerances for comparing two tensors."""
    fudge_factor = fudge_factor if fudge_factor is not None else 1.0
    atol = get_atol(true_value, computed_value)
    rtol = get_rtol(true_value, computed_value)

    atol = fudge_factor * max(atol, default_atol[computed_value.dtype])
    rtol = fudge_factor * max(rtol, default_rtol[computed_value.dtype])
    # torch.isclose() has weird behavior around see:
    # https://github.com/pytorch/pytorch/issues/102400
    if rtol > 1e30:
        rtol = default_rtol[computed_value.dtype]
    return atol, rtol


def query_key_value_clones(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dtype: torch.dtype = None):
    """ Clones the query, key, and value tensors and moves them to the specified dtype. """
    if dtype is None:
        dtype = query.dtype
    query_ref = query.clone().detach().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.clone().detach().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.clone().detach().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref


def rand_sdpa_tensor(shape: SdpaShape, device: str, dtype: torch.dtype, type: str,
                     requires_grad: bool = False, packed: bool = False) -> torch.Tensor:
    """Creates rand dense or nested tensor with given shape and type.

    Args:
        shape (Tuple[int]): Shape of Tensor to construct
        device (str): which device to create tensor on
        dtype (torch.dtype): Tensors' dtype
        type (str): Nested or Dense
        requires_grad (bool, optional): Tensors grad status. Defaults to False.
        packed (bool, optional): Whether to create a single QKV packed or not. Defaults to False.

    Returns:
        torch.Tensor: A new tensor
    """
    batch, num_heads, seq_len, head_dim = shape.batch, shape.num_heads, shape.seq_len, shape.head_dim
    if type == "nested":
        if isinstance(seq_len, list):
            def _size(i):
                return (seq_len[i], num_heads, head_dim) if not packed else (seq_len[i], 3 * num_heads * head_dim)

            return torch.nested.nested_tensor([
                torch.randn(_size(i), device=device, dtype=dtype, requires_grad=requires_grad)
                for i in range(batch)])
        else:
            size = (seq_len, num_heads, head_dim) if not packed else (seq_len, 3 * num_heads * head_dim)
            return torch.nested.nested_tensor([
                torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)
                for _ in range(batch)])
    else:
        assert (isinstance(seq_len, int))
        size = (batch, seq_len, num_heads, head_dim) if not packed else (batch, seq_len, 3 * num_heads * head_dim)
        return torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)

def calculate_nt_tolerances(nt_ref_hp, nt_ref_lp, default_dtype, fudge_factor=1):
    # TODO use NT ops when we have implemented Max for NestedTensor instead of unrolling
    ref_atol = default_atol[default_dtype]
    ref_rtol = default_rtol[default_dtype]
    for tensor_component_ref, tensor_component_ref_lp in zip(nt_ref_hp.unbind(), nt_ref_lp.unbind()):
        ref_atol = max((fudge_factor * torch.abs(tensor_component_ref - tensor_component_ref_lp)).max().item(), ref_atol)
        ref_rtol = max(get_rtol(tensor_component_ref, tensor_component_ref_lp), ref_rtol)
    return ref_atol, ref_rtol


# @pytest.fixture(scope='class', autouse=True)
# def set_env_var():
#     os.environ['USE_ONEDNN_GRAPH'] = "1"
#     yield
#     del os.environ['USE_ONEDNN_GRAPH']
class TestoneDNNGraphSDPAOnly(NNTestCase):
    """ Used to test oneDNN Graph fused SDPA only functionality of scaled_dot_product_attention
    """

    @pytest.fixture(scope='class', autouse=True)
    def device(self):
        return 'xpu:0'

    # @unittest.skip("OneDNN Graph does not support different dk dv")
    # def test_onednn_attention_different_dk_dv(self, device):
    #     dtype = torch.bfloat16
    #     make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
    #     batch, num_heads, head_dim_k, head_dim_v = 32, 16, 128, 64
    #     seq_len = 640
    #     q_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
    #     k_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
    #     v_shape = SdpaShape(batch, num_heads, seq_len, head_dim_v)
    #     query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

    #     with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
    #         actual = torch.nn.functional.scaled_dot_product_attention(
    #             query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
    #     with sdpa_kernel(backends=[SDPBackend.MATH]):
    #         math_ref = torch.nn.functional.scaled_dot_product_attention(
    #             query.contiguous().to(torch.float32),
    #             key.contiguous().to(torch.float32),
    #             value.contiguous().to(torch.float32),
    #             attn_mask=None, dropout_p=0.0, is_causal=False)

    #     self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)


    @parametrize("type", ["dense"])
    @parametrize("is_contiguous", [True, False])
    def test_scaled_dot_product_attention_fused_kernels_packed(self, device, type: str, is_contiguous: bool):
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=torch.float16, packed=True)

        batch_size, seq_len, num_heads, head_dim = 32, 64, 16, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)

        # Test Packed
        qkv = make_tensor(shape)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if is_contiguous:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        os.environ['USE_ONEDNN_GRAPH'] = "1"
        actual = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        del os.environ['USE_ONEDNN_GRAPH']

        math_ref = torch.nn.functional.scaled_dot_product_attention(
            query.contiguous(), key.contiguous(), value.contiguous(),
            attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=2e-3, rtol=1e-2)

    @parametrize("type", ["dense"])
    def test_scaled_dot_product_attention_fused_kernels_packed_accuracy(self, device, type: str):
        def rand_nt(shape):
            batch, seq_len, num_heads, head_dim = shape
            tensors = [6 * torch.rand((seq_len, 3 * num_heads * head_dim), device=device, dtype=torch.float32) - 3
                       for _ in range(batch)]
            return (torch.nested.nested_tensor(tensors, device=device, dtype=torch.float32),
                    torch.nested.nested_tensor(tensors, device=device, dtype=torch.float16))

        def rand_tensor(shape):
            batch, seq_len, num_heads, head_dim = shape
            tensor = 6 * torch.rand((batch, seq_len, 3 * num_heads * head_dim), device=device, dtype=torch.float32) - 3
            return tensor, tensor.to(dtype=torch.float16)

        batch_size, seq_len, num_heads, head_dim = 16, 8, 4, 64
        shape = (batch_size, seq_len, num_heads, head_dim)

        # Test Packed
        qkv, qkv_low_precision = rand_tensor(shape) if type == "dense" else rand_nt(shape)
        query, key, value = qkv.chunk(3, dim=-1)
        query_lp, key_lp, value_lp = qkv_low_precision.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query_lp = query_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_lp = key_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_lp = value_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        os.environ['USE_ONEDNN_GRAPH'] = "1"
        actual = torch.nn.functional.scaled_dot_product_attention(
            query_lp, key_lp, value_lp, attn_mask=None, dropout_p=0.0, is_causal=False)
        del os.environ['USE_ONEDNN_GRAPH']

        math_ref_lp = torch.nn.functional.scaled_dot_product_attention(
            query_lp.contiguous(), key_lp.contiguous(), value_lp.contiguous(),
            attn_mask=None, dropout_p=0.0, is_causal=False)

        math_query = query.contiguous()
        math_key = key.contiguous()
        math_value = value.contiguous()

        math_ref = torch.nn.functional.scaled_dot_product_attention(
            math_query, math_key, math_value, attn_mask=None, dropout_p=0.0, is_causal=False)

        actual_test = actual
        math_ref_test = math_ref
        math_ref_lp_test = math_ref_lp

        if actual_test.is_nested:
            actual_test = torch.nested.to_padded_tensor(actual_test.contiguous(), padding=0.0)
            math_ref_test = torch.nested.to_padded_tensor(math_ref_test, padding=0.0)
            math_ref_lp_test = torch.nested.to_padded_tensor(math_ref_lp_test, padding=0.0)

        actual_test = actual_test.to(dtype=torch.float32).contiguous()
        math_ref_test = math_ref_test.to(dtype=torch.float32).contiguous()
        math_ref_lp_test = math_ref_lp_test.to(dtype=torch.float32).contiguous()

        self.assertEqual(math_ref_test, math_ref_lp_test, atol=8e-3, rtol=7e-3)
        self.assertEqual(actual_test, math_ref_test, atol=7e-3, rtol=7e-3)


    # @parametrize("dtype", [torch.half])
    # @parametrize("batch_size", [4])
    # @parametrize("q_seq_len", [12])
    # @parametrize("kv_seq_len", [12])
    # @parametrize("q_head", [32])
    # @parametrize("kv_head", [8])
    # @parametrize("head_dim", [64])
    # @parametrize("mask_type", ["none", "float"])
    # @parametrize("train", [False])
    # def test_scaled_dot_product_attention_mask_vs_math_gqa(
    #     self,
    #     device,
    #     dtype,
    #     batch_size,
    #     q_seq_len,
    #     kv_seq_len,
    #     q_head,
    #     kv_head,
    #     head_dim,
    #     mask_type,
    #     train,
    # ):
    #     tol = Tolerances(1e-5, 5e-6)
    #     if dtype is torch.bfloat16:
    #         tol = Tolerances(5e-2, 5e-2)
    #     if dtype is torch.float16:
    #         tol = Tolerances(1e-2, 1e-2)
    #     mask_shape = [batch_size, 1, 1, kv_seq_len]
    #     make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype, requires_grad=False)
    #     q_shape = SdpaShape(batch_size, q_head, q_seq_len, head_dim)
    #     kv_shape = SdpaShape(batch_size, kv_head, kv_seq_len, head_dim)
    #     q = make_tensor(q_shape)
    #     k = make_tensor(kv_shape)
    #     v = make_tensor(kv_shape)
    #     q[:] = 0
    #     k[:] = 0
    #     q2, k2, v2 = q.clone(), k.clone(), v.clone()

    #     if train:
    #         q.requires_grad_(True)
    #         k.requires_grad_(True)
    #         v.requires_grad_(True)
    #         q2.requires_grad_(True)
    #         k2.requires_grad_(True)
    #         v2.requires_grad_(True)

    #     # (B, nh, T, hs)
    #     q = q.view(batch_size, q_seq_len, q_head, head_dim).transpose(1, 2)
    #     k = k.view(batch_size, kv_seq_len, kv_head, head_dim).transpose(1, 2)
    #     v = v.view(batch_size, kv_seq_len, kv_head, head_dim).transpose(1, 2)
    #     attn_mask = None
    #     if mask_type == "bool":
    #         attn_mask = torch.randint(0, 2, size=mask_shape, dtype=torch.bool, device=device)
    #     elif mask_type == "float":
    #         attn_mask = torch.randn(mask_shape, dtype=dtype, device=device)

    #     q2 = q2.view(batch_size, q_seq_len, q_head, head_dim).transpose(1, 2)
    #     k2 = k2.view(batch_size, kv_seq_len, kv_head, head_dim).transpose(1, 2)
    #     v2 = v2.view(batch_size, kv_seq_len, kv_head, head_dim).transpose(1, 2)

    #     # waiting for "enalbe_gqa" flag support in PT2.5
    #     # with sdpa_kernel(backends=[fused_kernel]):
    #     #     actual = torch.nn.functional.scaled_dot_product_attention(
    #     #         q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, enable_gqa=True)
    #     # with sdpa_kernel(backends=[SDPBackend.MATH]):
    #     #     math_ref = torch.nn.functional.scaled_dot_product_attention(
    #     #         q2, k2, v2, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, enable_gqa=True)

    #     os.environ['USE_ONEDNN_GRAPH'] = "1"
    #     actual = torch.nn.functional.scaled_dot_product_attention(
    #         q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    #     del os.environ['USE_ONEDNN_GRAPH']

    #     math_ref = torch.nn.functional.scaled_dot_product_attention(
    #         q2, k2, v2, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    #     if dtype in [torch.bfloat16, torch.float16]:
    #         math_ref = math_ref.to(dtype)

    #     self.assertEqual(actual, math_ref, atol=tol.atol, rtol=tol.rtol)


    # @parametrize("dtype", [torch.half])
    # @parametrize("batch_size", [4])
    # @parametrize("q_seq_len", [12])
    # @parametrize("kv_seq_len", [12])
    # @parametrize("q_head", [32])
    # @parametrize("kv_head", [8])
    # @parametrize("head_dim", [64])
    # @parametrize("mask_type", ["none", "float"])
    # @parametrize("train", [False])
    # def test_scaled_dot_product_attention_mask_gqa_cp_cache(
    #     self,
    #     device,
    #     dtype,
    #     batch_size,
    #     q_seq_len,
    #     kv_seq_len,
    #     q_head,
    #     kv_head,
    #     head_dim,
    #     mask_type,
    #     train,
    # ):
    #     tol = Tolerances(1e-5, 5e-6)
    #     if dtype is torch.bfloat16:
    #         tol = Tolerances(5e-2, 5e-2)
    #     if dtype is torch.float16:
    #         tol = Tolerances(1e-2, 1e-2)
    #     mask_shape = [batch_size, 1, 1, kv_seq_len]
    #     make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype, requires_grad=False)
    #     q_shape = SdpaShape(batch_size, q_head, q_seq_len, head_dim)
    #     kv_shape = SdpaShape(batch_size, kv_head, kv_seq_len, head_dim)
    #     q = make_tensor(q_shape)
    #     k = make_tensor(kv_shape)
    #     v = make_tensor(kv_shape)
    #     q[:] = 0
    #     k[:] = 0

    #     if train:
    #         q.requires_grad_(True)
    #         k.requires_grad_(True)
    #         v.requires_grad_(True)

    #     # (B, nh, T, hs)
    #     q = q.view(batch_size, q_seq_len, q_head, head_dim).transpose(1, 2)
    #     k = k.view(batch_size, kv_seq_len, kv_head, head_dim).transpose(1, 2)
    #     v = v.view(batch_size, kv_seq_len, kv_head, head_dim).transpose(1, 2)
    #     attn_mask = None
    #     if mask_type == "bool":
    #         attn_mask = torch.randint(0, 2, size=mask_shape, dtype=torch.bool, device=device)
    #     elif mask_type == "float":
    #         attn_mask = torch.randn(mask_shape, dtype=dtype, device=device)

    #     # waiting for "enalbe_gqa" flag support in PT2.5
    #     # with sdpa_kernel(backends=[fused_kernel]):
    #     #     actual = torch.nn.functional.scaled_dot_product_attention(
    #     #         q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, enable_gqa=True)
    #     os.environ['USE_ONEDNN_GRAPH'] = "1"
    #     actual = torch.nn.functional.scaled_dot_product_attention(
    #         q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    #     del os.environ['USE_ONEDNN_GRAPH']

    @parametrize("dtype", [torch.half])
    @parametrize("batch_size", [4])
    @parametrize("q_seq_len", [1024])
    @parametrize("kv_seq_len", [1024])
    @parametrize("n_head", [32])
    @parametrize("head_dim", [128])
    @parametrize("mask_type", ["none", "float"])
    @parametrize("train", [False])
    def test_scaled_dot_product_attention_mask_vs_math(
        self,
        device,
        dtype,
        batch_size,
        q_seq_len,
        kv_seq_len,
        n_head,
        head_dim,
        mask_type,
        train,
    ):
        tol = Tolerances(1e-5, 5e-6)
        if dtype is torch.bfloat16:
            tol = Tolerances(5e-2, 5e-2)
        if dtype is torch.float16:
            tol = Tolerances(1e-2, 1e-2)
        mask_shape = [batch_size, 1, 1, kv_seq_len]
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype, requires_grad=False)
        q_shape = SdpaShape(batch_size, n_head, q_seq_len, head_dim)
        kv_shape = SdpaShape(batch_size, n_head, kv_seq_len, head_dim)
        q = make_tensor(q_shape)
        k = make_tensor(kv_shape)
        v = make_tensor(kv_shape)
        q2, k2, v2 = q.clone(), k.clone(), v.clone()

        if train:
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            q2.requires_grad_(True)
            k2.requires_grad_(True)
            v2.requires_grad_(True)

        # (B, nh, T, hs)
        q = q.view(batch_size, q_seq_len, n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, n_head, head_dim).transpose(1, 2)
        attn_mask = None
        if mask_type == "bool":
            attn_mask = torch.randint(0, 2, size=mask_shape, dtype=torch.bool, device=device)
        elif mask_type == "float":
            attn_mask = torch.randn(mask_shape, dtype=dtype, device=device)

        q2 = q2.view(batch_size, q_seq_len, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(batch_size, kv_seq_len, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, kv_seq_len, n_head, head_dim).transpose(1, 2)

        os.environ['USE_ONEDNN_GRAPH'] = "1"
        actual = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        del os.environ['USE_ONEDNN_GRAPH']

        math_ref = torch.nn.functional.scaled_dot_product_attention(
            q2, k2, v2, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

        if dtype in [torch.bfloat16, torch.float16]:
            math_ref = math_ref.to(dtype)

        self.assertEqual(actual, math_ref, atol=tol.atol, rtol=tol.rtol)


# instantiate_device_type_tests(
#     TestoneDNNGraphSDPAOnly, globals(), only_for="xpu")
instantiate_device_type_tests(
    TestoneDNNGraphSDPAOnly, globals())

if __name__ == "__main__":
    common.xpu_test_base.customized_skipper()
    run_tests()
