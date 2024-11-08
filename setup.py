import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, _find_cuda_home
import torch
import platform

CPU_COUNT = os.cpu_count()
generator_flag = []
torch_dir = torch.__path__[0]

cc_flag = []


def find_cublas_headers():
    home = _find_cuda_home()
    if home is None:
        raise EnvironmentError(
            "CUDA environment not found, ensure that you have CUDA toolkit installed locally, and have added it to your environment variables as CUDA_HOME=/path/to/cuda-12.x"
        )
    if platform.system() == "Windows":
        cublas_include = os.path.join(home, "include")
        cublas_libs = os.path.join(home, "lib", "x64")
    else:
        cublas_include = os.path.join(home, "include")
        cublas_libs = os.path.join(home, "lib64")

    return cublas_include, cublas_libs


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or f"{min(CPU_COUNT, 8)}"
    return nvcc_extra_args + ["--threads", nvcc_threads]


setup(
    name="f8_fast_matmul",
    version="0.1.1",
    ext_modules=[
        CUDAExtension(
            name="f8_fast_matmul_ext",
            sources=[
                "src/extension.cpp",
                "src/quantize_fp8.cu",
                "src/quantize_fp8_experimental.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                    ]
                    + generator_flag
                    + cc_flag
                ),
            },
            include_dirs=[*find_cublas_headers()],
        ),
    ],
    packages=find_packages(
        exclude=[
            ".misc",
            "__pycache__",
            ".vscode",
            "cublas_ops.egg-info",
            "build",
            "ttt",
        ]
    ),
    cmdclass={"build_ext": BuildExtension},
)
