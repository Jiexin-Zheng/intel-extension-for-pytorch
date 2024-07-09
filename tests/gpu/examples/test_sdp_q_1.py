import torch
import intel_extension_for_pytorch  # noqa
import torch.nn.functional as F

from torch.testing._internal.common_utils import TestCase
import pytest

checking_atol = 1e-3
checking_rtol = 1e-3


class TestTorchMethod(TestCase):
    # not support on DG2 yet
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(), reason="fallback is required"
    )
    def test_fsdp_base(self, dtype=torch.float16):
        head_dim = 256
        seq_lenth = 1
        k_seq_lenth = 33
        v_seq_lenth = 33
        query = torch.rand(1, 16, seq_lenth, head_dim, dtype=dtype)
        key = torch.rand(1, 16, k_seq_lenth, head_dim, dtype=dtype)
        value = torch.rand(1, 16, v_seq_lenth, head_dim, dtype=dtype)

        out_cpu = F.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), is_causal=False
        )
        out_xpu = F.scaled_dot_product_attention(
            query.xpu(), key.xpu(), value.xpu(), is_causal=False
        )

        self.assertEqual(
            out_cpu, out_xpu.cpu().float(), atol=checking_atol, rtol=checking_rtol
        )

    # not support on DG2 yet
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(), reason="fallback is required"
    )
    def test_fsdp_strided(self, dtype=torch.float16):
        query = torch.permute(
            torch.reshape(torch.rand(1, 1, 4096), (1, 1, 16, 256)), (0, 2, 1, 3)
        ).half()
        key = torch.permute(
            torch.reshape(torch.rand(1, 2048, 4096), (1, 2048, 16, 256)), (0, 2, 1, 3)
        )[:, :, :33, :].half()
        value = torch.permute(
            torch.reshape(torch.rand(1, 2048, 4096), (1, 2048, 16, 256)), (0, 2, 1, 3)
        )[:, :, :33, :].half()

        out_cpu = F.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), is_causal=False
        )
        out_xpu = F.scaled_dot_product_attention(
            query.xpu(), key.xpu(), value.xpu(), is_causal=False
        )

        self.assertEqual(
            out_cpu, out_xpu.cpu().float(), atol=checking_atol, rtol=checking_rtol
        )

    # not support on DG2 yet
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(), reason="fallback is required"
    )
    def test_fsdp_causal(self, dtype=torch.float16):
        head_dim = 256
        seq_lenth = 1
        k_seq_lenth = 33
        v_seq_lenth = 33
        query = torch.rand(1, 16, seq_lenth, head_dim, dtype=dtype)
        key = torch.rand(1, 16, k_seq_lenth, head_dim, dtype=dtype)
        value = torch.rand(1, 16, v_seq_lenth, head_dim, dtype=dtype)

        out_cpu = F.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), is_causal=True
        )
        out_xpu = F.scaled_dot_product_attention(
            query.xpu(), key.xpu(), value.xpu(), is_causal=True
        )

        self.assertEqual(
            out_cpu, out_xpu.cpu().float(), atol=checking_atol, rtol=checking_rtol
        )

    # not support on DG2 yet
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(), reason="fallback is required"
    )
    def test_fsdp_causal_stride(self, dtype=torch.float16):
        query = torch.permute(
            torch.reshape(torch.rand(1, 1, 4096), (1, 1, 16, 256)), (0, 2, 1, 3)
        ).half()
        key = torch.permute(
            torch.reshape(torch.rand(1, 2048, 4096), (1, 2048, 16, 256)), (0, 2, 1, 3)
        )[:, :, :33, :].half()
        value = torch.permute(
            torch.reshape(torch.rand(1, 2048, 4096), (1, 2048, 16, 256)), (0, 2, 1, 3)
        )[:, :, :33, :].half()

        out_cpu = F.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), is_causal=True
        )
        out_xpu = F.scaled_dot_product_attention(
            query.xpu(), key.xpu(), value.xpu(), is_causal=True
        )

        self.assertEqual(
            out_cpu, out_xpu.cpu().float(), atol=checking_atol, rtol=checking_rtol
        )
