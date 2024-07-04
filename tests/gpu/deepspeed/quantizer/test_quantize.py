import torch
import pytest
import intel_extension_for_pytorch as ipex

quantizer_xpu_module = ipex.xpu.deepspeed
ipex_device = 'xpu:0'

def run_quantize_ds(activations, num_groups, q_bits, is_symmetric_quant):

    return quantizer_xpu_module.quantize(activations, num_groups, q_bits,
                                     is_symmetric_quant)


def run_dequantize_ds(activations, params, num_groups, q_bits, is_symmetric_quant):
    return quantizer_xpu_module.dequantize(
        activations,
        params,
        num_groups,
        q_bits,
        is_symmetric_quant
    )


def get_q_props(q_bits):
    q_range = 2**q_bits
    q_min = -(2**(q_bits - 1))
    q_max = (2**(q_bits - 1) - 1)

    q_min = torch.IntTensor([q_min]).to(device=ipex_device)
    q_max = torch.IntTensor([q_max]).to(device=ipex_device)
    return q_range, q_max, q_min


def get_scale_zero_point(q_bits, is_symmetric_quant, max, min, absmax, scales=None, zero_points=None):

    q_range, q_max, q_min = get_q_props(q_bits)

    if is_symmetric_quant:
        scale = torch.empty_like(absmax)
        for i, x in enumerate(absmax):
            scale[i] = torch.ones_like(x) if x == 0 else q_range / (2 * x)
        zero_point = torch.zeros(scale.shape, dtype=torch.float32, device=ipex_device)
    else:
        scale = torch.empty_like(max)
        for i, x in enumerate(max):
            scale[i] = torch.ones_like(x) if max[i] == min[i] else q_range / (max[i] - min[i])
        zero_point = q_min - (min * scale)

    return scale, zero_point


def int4x2to2xint4(int4X2tensor):
    high = int4X2tensor >> 4
    low = (int4X2tensor << 4) >> 4
    return torch.stack((high, low), dim=-1).flatten()


def run_float_quantize(q_bits, is_symmetric_quant, activations_ref, num_groups):

    # Reference implementation
    # https://pytorch.org/docs/stable/quantization-support.html

    activations_ref = activations_ref.reshape(num_groups, -1).to(dtype=torch.float32)

    max_abs_activations_ref = torch.amax(torch.abs(activations_ref), dim=-1).view(num_groups, -1)
    max_activations_ref = torch.amax(activations_ref, dim=-1).view(num_groups, -1)
    min_activations_ref = torch.amin(activations_ref, dim=-1).view(num_groups, -1)

    _, q_max, q_min = get_q_props(q_bits)

    scale, zero_point = get_scale_zero_point(q_bits, is_symmetric_quant, max_activations_ref, min_activations_ref,
                                             max_abs_activations_ref)

    data_f = activations_ref * scale

    if not is_symmetric_quant:
        data_f = data_f + zero_point

    data_i32 = torch.round(data_f).to(dtype=torch.int32)

    data_i32 = torch.minimum(torch.maximum(data_i32, q_min.expand_as(data_i32)), q_max.expand_as(data_i32))
    data_i8 = data_i32.to(dtype=torch.int8)

    scales = (1.0 / scale).reshape(-1, 1)
    offsets = zero_point.reshape(-1, 1)
    params = torch.cat((scales, offsets), dim=-1)

    return data_i8, params


def run_float_dequantize(q_bits, is_symmetric_quant, data_i8, params, num_groups):
    data_f = data_i8.reshape(num_groups, -1).to(dtype=torch.float32)

    scales = params[:, 0].reshape(-1, 1)
    offsets = params[:, 1].reshape(-1, 1)

    if not is_symmetric_quant:
        data_f = data_f - offsets
    else:
        assert offsets.allclose(torch.zeros_like(offsets))

    data_f = data_f * scales

    return data_f


@pytest.mark.skipif(not quantizer_xpu_module.has_deepspeed(), reason="deepspeed module is not available")
@pytest.mark.parametrize("num_groups", [1, 13, 512])
@pytest.mark.parametrize("num_elems", [8, 16, 32, 64, 128, 4096, 8192, 12288, 16384])
@pytest.mark.parametrize("is_symmetric_quant", [True, False])
@pytest.mark.parametrize("q_bits", [4, 8])
@pytest.mark.parametrize("directed_case", ["all_zeros", None])
def test_float_quantize(num_elems, num_groups, is_symmetric_quant, q_bits, directed_case):
    # fix seed
    torch.manual_seed(num_elems)

    if directed_case == "all_zeros":
        activations_ds = torch.zeros((num_groups, num_elems),
                                     dtype=torch.float16,
                                     device=ipex_device)
    else:
        activations_ds = torch.randn((num_groups, num_elems),
                                     dtype=torch.float16,
                                     device=ipex_device)
    activations_ref = activations_ds.clone().detach()

    ref_out_tensor, ref_params = run_float_quantize(q_bits, is_symmetric_quant, activations_ref, num_groups)
    ref_dequantized_tensor = run_float_dequantize(q_bits, is_symmetric_quant, ref_out_tensor, ref_params, num_groups)
    # we need to convert the tensor to float64 to avoid overflow
    ref_quantization_error = torch.sum(torch.abs((activations_ref - ref_dequantized_tensor).to(torch.float64)))

    ds_out_tensor, ds_out_params = run_quantize_ds(activations_ds, num_groups, q_bits, is_symmetric_quant)
    ds_dequantized_tensor = run_dequantize_ds(ds_out_tensor, ds_out_params, num_groups, q_bits, is_symmetric_quant)
    assert torch.all(torch.isfinite(ds_dequantized_tensor))

    ds_quantization_error = torch.sum(torch.abs((activations_ds - ds_dequantized_tensor).to(torch.float64)))

    assert (ds_quantization_error <= ref_quantization_error * 1.05)
