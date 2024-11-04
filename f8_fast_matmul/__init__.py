import torch
from f8_fast_matmul_ext import quantize_fp8_forward as _quantize_fp8_forward, quantize_fp8_forward_experimental as _quantize_fp8_forward_experimental  # type: ignore


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
    n_load: int = 16,
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
