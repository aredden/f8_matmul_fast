#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#define CDIV(a, b) ((a) + (b) - 1) / (b)
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
    default:
        return __nv_cvt_float_to_fp8(fvalue, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
    };
}

__device__ __forceinline__ __nv_fp8x2_storage_t convert_to_fp8_2(float2 fvalue, int f8_dtype)
{
    switch (f8_dtype)
    {
    case 0:
        return reinterpret_cast<__nv_fp8x2_storage_t>(__nv_cvt_float2_to_fp8x2(
            fvalue,
            __nv_saturation_t::__NV_SATFINITE,
            __nv_fp8_interpretation_t::__NV_E5M2));
    case 1:
        return reinterpret_cast<__nv_fp8x2_storage_t>(__nv_cvt_float2_to_fp8x2(
            fvalue,
            __nv_saturation_t::__NV_SATFINITE,
            __nv_fp8_interpretation_t::__NV_E4M3));
    };
}

__device__ __forceinline__ __nv_fp8x4_storage_t convert_to_fp8_4(float4 fvalue, int f8_dtype, __nv_fp8x2_storage_t *temp_storage)
{
    switch (f8_dtype)
    {
    case 0:
        temp_storage[0] = __nv_cvt_float2_to_fp8x2(make_float2(fvalue.x, fvalue.y), __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E5M2);
        temp_storage[1] = __nv_cvt_float2_to_fp8x2(make_float2(fvalue.z, fvalue.w), __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E5M2);
        return reinterpret_cast<__nv_fp8x4_storage_t *>(temp_storage)[0];
    case 1:
        temp_storage[0] = __nv_cvt_float2_to_fp8x2(make_float2(fvalue.x, fvalue.y), __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
        temp_storage[1] = __nv_cvt_float2_to_fp8x2(make_float2(fvalue.z, fvalue.w), __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
        return reinterpret_cast<__nv_fp8x4_storage_t *>(temp_storage)[0];
    };
}

template <typename scalar_t>
__global__ void quantize_fp8_forward_experimental(
    const scalar_t *input,
    __nv_fp8_storage_t *output,
    const float scale,
    const float max_value,
    const uint32_t M,
    const uint32_t N,
    const uint32_t stride_m,
    const uint32_t stride_n,
    const int f8_dtype,
    const int n_load)
{

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t offset_x = (i % M);
    uint32_t offset_y = (i / M);

    uint32_t output_offset = (offset_y * M + offset_x);

    scalar_t values[8];
    __nv_fp8x2_storage_t temp_storage[4];
    if (output_offset * n_load < M * N)
    {
        if (n_load == 1)
        {
            scalar_t value = input[output_offset];
            float quantized_value;
            quantized_value = fminf(fmaxf(float(value) * scale, -max_value), max_value);
            output[output_offset] = convert_to_fp8(quantized_value, f8_dtype);
        }
        else if (n_load == 2)
        {
            // load 2 values, since half precision
            reinterpret_cast<float *>(values)[0] = reinterpret_cast<const float *>(input)[output_offset];
            float quantized_value_0 = fminf(fmaxf(float(values[0]) * scale, -max_value), max_value);
            float quantized_value_1 = fminf(fmaxf(float(values[1]) * scale, -max_value), max_value);
            reinterpret_cast<__nv_fp8x2_storage_t *>(output)[output_offset] = convert_to_fp8_2(make_float2(quantized_value_0, quantized_value_1), f8_dtype);
        }
        else if (n_load == 4)
        {
            // load 4 values, since float precision
            reinterpret_cast<float2 *>(values)[0] = const_cast<float2 *>(reinterpret_cast<const float2 *>(input))[output_offset];
            float quantized_value_0 = fminf(fmaxf(float(values[0]) * scale, -max_value), max_value);
            float quantized_value_1 = fminf(fmaxf(float(values[1]) * scale, -max_value), max_value);
            float quantized_value_2 = fminf(fmaxf(float(values[2]) * scale, -max_value), max_value);
            float quantized_value_3 = fminf(fmaxf(float(values[3]) * scale, -max_value), max_value);
            reinterpret_cast<__nv_fp8x4_storage_t *>(output)[output_offset] = convert_to_fp8_4(make_float4(quantized_value_0, quantized_value_1, quantized_value_2, quantized_value_3), f8_dtype, temp_storage);
        }
        else if (n_load > 4)
        {
            // assume n_load is a multiple of 8
            for (int j = 0; j < n_load / 8; j += 1)
            {
                reinterpret_cast<float4 *>(values)[0] = const_cast<float4 *>(reinterpret_cast<const float4 *>(input))[(output_offset * (n_load / 8)) + j];
                float quantized_value_0 = fminf(fmaxf(float(values[0]) * scale, -max_value), max_value);
                float quantized_value_1 = fminf(fmaxf(float(values[1]) * scale, -max_value), max_value);
                float quantized_value_2 = fminf(fmaxf(float(values[2]) * scale, -max_value), max_value);
                float quantized_value_3 = fminf(fmaxf(float(values[3]) * scale, -max_value), max_value);
                float quantized_value_4 = fminf(fmaxf(float(values[4]) * scale, -max_value), max_value);
                float quantized_value_5 = fminf(fmaxf(float(values[5]) * scale, -max_value), max_value);
                float quantized_value_6 = fminf(fmaxf(float(values[6]) * scale, -max_value), max_value);
                float quantized_value_7 = fminf(fmaxf(float(values[7]) * scale, -max_value), max_value);
                temp_storage[0] = convert_to_fp8_2(make_float2(quantized_value_0, quantized_value_1), f8_dtype);
                temp_storage[1] = convert_to_fp8_2(make_float2(quantized_value_2, quantized_value_3), f8_dtype);
                temp_storage[2] = convert_to_fp8_2(make_float2(quantized_value_4, quantized_value_5), f8_dtype);
                temp_storage[3] = convert_to_fp8_2(make_float2(quantized_value_6, quantized_value_7), f8_dtype);
                reinterpret_cast<float2 *>(output)[(output_offset * (n_load / 8)) + j] = reinterpret_cast<float2 *>(temp_storage)[0];
            }
        }
    }
}

torch::Tensor quantize_fp8_forward_experimental(
    const torch::Tensor input,
    const float scale,
    const float max_value,
    const int f8_dtype,
    const int n_load,
    const int block_size)
{
    const uint32_t M = input.size(0);
    const uint32_t N = input.size(1);

    const uint32_t stride_m = input.stride(0);
    const uint32_t stride_n = input.stride(1);

    TORCH_CHECK(M > 0 && N > 0, "Illegal input size, must be greater than 0");
    TORCH_CHECK(scale > 0, "Illegal scale, must be greater than 0");
    TORCH_CHECK(max_value > 0, "Illegal max_value, must be greater than 0");
    TORCH_CHECK(f8_dtype == 0 || f8_dtype == 1, "Illegal f8_dtype, must be 0 or 1 (Float8_e5m2 or Float8_e4m3fn)");
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(n_load == 1 || n_load == 2 || n_load == 4 || (n_load % 4 == 0 && n_load > 4), "Illegal n_load, must be 1, 2, or 4");
    torch::Tensor output;
    int grid_size;

    if (f8_dtype == 0)
    {
        output = torch::empty_like(input, torch::TensorOptions().device(input.device()).dtype(torch::ScalarType::Float8_e5m2));
    }
    else
    {
        output = torch::empty_like(input, torch::TensorOptions().device(input.device()).dtype(torch::ScalarType::Float8_e4m3fn));
    }
    grid_size = (M * N + block_size - 1) / (block_size * n_load);

    if (input.scalar_type() == torch::ScalarType::Half)
    {
        quantize_fp8_forward_experimental<__half><<<grid_size, block_size>>>(
            reinterpret_cast<const __half *>(input.const_data_ptr()),
            reinterpret_cast<__nv_fp8_storage_t *>(output.mutable_data_ptr()),
            scale,
            max_value,
            M,
            N,
            stride_m,
            stride_n,
            f8_dtype,
            n_load);
    }
    else if (input.scalar_type() == torch::ScalarType::BFloat16)
    {
        quantize_fp8_forward_experimental<__nv_bfloat16><<<grid_size, block_size>>>(
            reinterpret_cast<const __nv_bfloat16 *>(input.const_data_ptr()),
            reinterpret_cast<__nv_fp8_storage_t *>(output.mutable_data_ptr()),
            scale,
            max_value,
            M,
            N,
            stride_m,
            stride_n,
            f8_dtype,
            n_load);
    }

    return output;
}