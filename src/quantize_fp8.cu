#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
// offset_x = (pid * block_h) % M
// offset_y = ((pid * block_w) // N) * block_w

__device__ __forceinline__ __nv_fp8_storage_t convert_to_fp8(float fvalue, int f8_dtype)
{
    switch (f8_dtype)
    {
    case 0:
        return __nv_cvt_float_to_fp8(fvalue, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E5M2);
    case 1:
        return __nv_cvt_float_to_fp8(fvalue, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
    };
}

template <typename scalar_t>
__global__ void quantize_fp8_forward(
    const scalar_t *input,
    __nv_fp8_storage_t *output,
    float scale,
    float max_value,
    uint32_t M,
    uint32_t N,
    uint32_t stride_m,
    uint32_t stride_n,
    int f8_dtype)
{

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t offset_x = (i % M);
    uint32_t offset_y = (i / M);

    uint32_t output_offset = offset_y * M + offset_x;

    if (i < M * N)
    {
        scalar_t value = input[output_offset];
        float quantized_value;
        quantized_value = fminf(fmaxf(float(value) * scale, -max_value), max_value);
        output[output_offset] = convert_to_fp8(quantized_value, f8_dtype);
    }
}

torch::Tensor quantize_fp8_forward(
    const torch::Tensor input,
    float scale,
    float max_value,
    int f8_dtype)
{
    uint32_t M = input.size(0);
    uint32_t N = input.size(1);

    uint32_t stride_m = input.stride(0);
    uint32_t stride_n = input.stride(1);

    TORCH_CHECK(M > 0 && N > 0, "Illegal input size, must be greater than 0");
    TORCH_CHECK(scale > 0, "Illegal scale, must be greater than 0");
    TORCH_CHECK(max_value > 0, "Illegal max_value, must be greater than 0");
    TORCH_CHECK(f8_dtype == 0 || f8_dtype == 1, "Illegal f8_dtype, must be 0 or 1 (Float8_e5m2 or Float8_e4m3fn)");
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");

    torch::Tensor output;
    int grid_size;
    int block_size = 1024;

    output = torch::empty_like(input, torch::TensorOptions().device(input.device()).dtype(torch::ScalarType::Float8_e5m2));
    grid_size = (M * N + block_size - 1) / block_size;

    if (input.scalar_type() == torch::ScalarType::Half)
    {
        quantize_fp8_forward<__half><<<grid_size, block_size>>>(
            reinterpret_cast<__half *>(input.mutable_data_ptr()),
            reinterpret_cast<__nv_fp8_storage_t *>(output.mutable_data_ptr()),
            scale,
            max_value,
            M,
            N,
            stride_m,
            stride_n,
            f8_dtype);
    }
    else if (input.scalar_type() == torch::ScalarType::BFloat16)
    {
        quantize_fp8_forward<__nv_bfloat16><<<grid_size, block_size>>>(
            reinterpret_cast<__nv_bfloat16 *>(input.mutable_data_ptr()),
            reinterpret_cast<__nv_fp8_storage_t *>(output.mutable_data_ptr()),
            scale,
            max_value,
            M,
            N,
            stride_m,
            stride_n,
            f8_dtype);
    }

    return output;
}