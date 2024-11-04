## FP8 Fast Matmul

-   Though currently it only does quantization from [bf16/fp16] to fp8.
-   The code is in `src/quantize_fp8_experimental.cu`.
-   Seems like the fastest is block size 128, n_load 8.
