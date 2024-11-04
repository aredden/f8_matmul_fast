#include <cuda_runtime.h>
#include <torch/extension.h>

torch::Tensor quantize_fp8_forward(
    torch::Tensor input, float scale, float max_value, int f8_dtype);

torch::Tensor quantize_fp8_forward_experimental(
    torch::Tensor input, float scale, float max_value, int f8_dtype, int n_load, int block_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantize_fp8_forward", &quantize_fp8_forward, "Quantize FP8 forward");
    m.def("quantize_fp8_forward_experimental", &quantize_fp8_forward_experimental, "Quantize FP8 forward experimental");
}
