#include <cuda_runtime.h>
#include <torch/extension.h>

torch::Tensor quantize_fp8_forward(
    torch::Tensor input, float scale, float max_value, int f8_dtype);

torch::Tensor quantize_fp8_forward_experimental(
    torch::Tensor input, float scale, float max_value, int f8_dtype, int n_load, int block_size);

torch::Tensor matmul_qfloat8_tscales(
    const torch::Tensor a,
    const torch::Tensor b_q,
    const torch::Tensor a_scale,
    const torch::Tensor b_scale,
    const torch::Tensor a_max_value,
    const ::std::optional<at::Tensor> &bias = {},
    const ::std::optional<bool> use_fast_accum = false,
    const int f8_dtype = 0);

torch::Tensor matmul_qfloat8(
    const torch::Tensor a,
    const torch::Tensor b_q,
    const float a_scale,
    const float a_max_value,
    const float b_scale,
    const ::std::optional<at::Tensor> &bias = {},
    const ::std::optional<bool> use_fast_accum = false,
    const int f8_dtype = 0);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantize_fp8_forward", &quantize_fp8_forward, "Quantize FP8 forward");
    m.def("quantize_fp8_forward_experimental", &quantize_fp8_forward_experimental, "Quantize FP8 forward experimental");
    m.def("matmul_qfloat8_tscales", &matmul_qfloat8_tscales, "Matmul QFloat8 with tensor scales");
    m.def("matmul_qfloat8", &matmul_qfloat8, "Matmul QFloat8");
}
