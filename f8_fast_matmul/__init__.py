import torch
from f8_fast_matmul_ext import quantize_fp8_forward as _quantize_fp8_forward, quantize_fp8_forward_experimental as _quantize_fp8_forward_experimental  # type: ignore
from f8_fast_matmul_ext import matmul_qfloat8_tscales as _matmul_qfloat8_tscales
from f8_fast_matmul_ext import matmul_qfloat8 as _matmul_qfloat8


@torch.inference_mode()
def quantize_fp8_forward(
    input: torch.Tensor,
    scale: float,
    max_value: float,
    f8_dtype: torch.dtype,
    n_load: int = 1,
) -> torch.Tensor:
    if f8_dtype == torch.float8_e5m2:
        f8_dtype = 0
    elif f8_dtype == torch.float8_e4m3fn:
        f8_dtype = 1
    else:
        raise ValueError(
            "Illegal f8_dtype, must be 0 or 1 (Float8_e5m2 or Float8_e4m3fn)"
        )
    if input.ndim != 2:
        output_shape = input.shape
        input = input.view(-1, input.size(-1))
    else:
        output_shape = input.shape
    return _quantize_fp8_forward(input, scale, max_value, f8_dtype).reshape(
        *output_shape
    )


@torch.inference_mode()
def quantize_fp8_forward_experimental(
    input: torch.Tensor,
    scale: float,
    max_value: float,
    f8_dtype: torch.dtype,
    n_load: int = 8,
    block_size: int = 128,
) -> torch.Tensor:
    if f8_dtype == torch.float8_e5m2:
        f8_dtype = 0
    elif f8_dtype == torch.float8_e4m3fn:
        f8_dtype = 1
    else:
        raise ValueError(
            "Illegal f8_dtype, must be 0 or 1 (Float8_e5m2 or Float8_e4m3fn)"
        )
    if input.ndim != 2:
        output_shape = input.shape
        input = input.view(-1, input.size(-1))
    else:
        output_shape = input.shape
    return _quantize_fp8_forward_experimental(
        input, scale, max_value, f8_dtype, n_load, block_size
    ).reshape(*output_shape)


def _to_f8_dtype(f8_dtype: torch.dtype) -> int:
    if f8_dtype == torch.float8_e5m2:
        return 0
    elif f8_dtype == torch.float8_e4m3fn:
        return 1
    else:
        raise ValueError("Illegal f8_dtype")


""""
torch::Tensor matmul_qfloat8_tscales(
    const torch::Tensor a,
    const torch::Tensor b_q,
    const torch::Tensor a_scale,
    const torch::Tensor b_scale,
    const torch::Tensor a_max_value,
    const ::std::optional<at::Tensor> &bias = {},
    const ::std::optional<bool> use_fast_accum = false,
    const int f8_dtype = 0)
"""


@torch.inference_mode()
def matmul_qfloat8_tscales(
    a: torch.Tensor,
    b_q: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    max_val_a: torch.Tensor,
    bias: torch.Tensor | None = None,
    use_fast_accum: bool = False,
    f8_dtype_a: torch.dtype = torch.float8_e5m2,
) -> torch.Tensor:
    return _matmul_qfloat8_tscales(
        a,
        b_q,
        scale_a,
        max_val_a,
        scale_b,
        bias,
        use_fast_accum,
        _to_f8_dtype(f8_dtype_a),
    )


@torch.inference_mode()
def matmul_qfloat8(
    a: torch.Tensor,
    b_q: torch.Tensor,
    scale_a: float,
    max_val_a: float,
    scale_b: float,
    bias: torch.Tensor | None = None,
    use_fast_accum: bool = False,
    f8_dtype: torch.dtype = torch.float8_e5m2,
) -> torch.Tensor:
    return _matmul_qfloat8(
        a,
        b_q,
        scale_a,
        max_val_a,
        scale_b,
        bias,
        use_fast_accum,
        _to_f8_dtype(f8_dtype),
    )
