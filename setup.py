from setuptools import setup, Extension
import importlib.util
import os

if torch := importlib.util.find_spec("torch") is not None:
    from torch.utils import cpp_extension
    from torch import version as torch_version

extension_name = "exllamav2_ext"
precompile = "EXLLAMA_NOCOMPILE" not in os.environ
verbose = "EXLLAMA_VERBOSE" in os.environ
ext_debug = "EXLLAMA_EXT_DEBUG" in os.environ

if precompile and not torch:
    print(
        "cannot precompile unless torch is installed \
To explicitly JIT install run EXLLAMA_NOCOMPILE= pip install <xyz>"
    )

windows = os.name == "nt"

extra_cflags = ["/Ox"] if windows else ["-O3"]

if ext_debug:
    extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]

extra_cuda_cflags = ["-lineinfo", "-O3"]

if torch and torch_version.hip:
    extra_cuda_cflags += ["-DHIPBLAS_USE_HIP_HALF"]

extra_compile_args = {
    "cxx": extra_cflags,
    "nvcc": extra_cuda_cflags,
}

setup_kwargs = (
    {
        "ext_modules": [
            cpp_extension.CUDAExtension(
                extension_name,
                [
                    "exllamav2/exllamav2_ext/ext_bindings.cpp",
                    "exllamav2/exllamav2_ext/ext_cache.cpp",
                    "exllamav2/exllamav2_ext/ext_gemm.cpp",
                    "exllamav2/exllamav2_ext/ext_hadamard.cpp",
                    "exllamav2/exllamav2_ext/ext_norm.cpp",
                    "exllamav2/exllamav2_ext/ext_qattn.cpp",
                    "exllamav2/exllamav2_ext/ext_qmatrix.cpp",
                    "exllamav2/exllamav2_ext/ext_qmlp.cpp",
                    "exllamav2/exllamav2_ext/ext_quant.cpp",
                    "exllamav2/exllamav2_ext/ext_rope.cpp",
                    "exllamav2/exllamav2_ext/ext_stloader.cpp",
                    "exllamav2/exllamav2_ext/ext_sampling.cpp",
                    "exllamav2/exllamav2_ext/ext_element.cpp",
                    "exllamav2/exllamav2_ext/ext_tp.cpp",
                    "exllamav2/exllamav2_ext/cuda/graph.cu",
                    "exllamav2/exllamav2_ext/cuda/h_add.cu",
                    "exllamav2/exllamav2_ext/cuda/h_gemm.cu",
                    "exllamav2/exllamav2_ext/cuda/lora.cu",
                    "exllamav2/exllamav2_ext/cuda/pack_tensor.cu",
                    "exllamav2/exllamav2_ext/cuda/quantize.cu",
                    "exllamav2/exllamav2_ext/cuda/q_matrix.cu",
                    "exllamav2/exllamav2_ext/cuda/q_attn.cu",
                    "exllamav2/exllamav2_ext/cuda/q_mlp.cu",
                    "exllamav2/exllamav2_ext/cuda/q_gemm.cu",
                    "exllamav2/exllamav2_ext/cuda/rms_norm.cu",
                    "exllamav2/exllamav2_ext/cuda/head_norm.cu",
                    "exllamav2/exllamav2_ext/cuda/layer_norm.cu",
                    "exllamav2/exllamav2_ext/cuda/rope.cu",
                    "exllamav2/exllamav2_ext/cuda/cache.cu",
                    "exllamav2/exllamav2_ext/cuda/util.cu",
                    "exllamav2/exllamav2_ext/cuda/softcap.cu",
                    "exllamav2/exllamav2_ext/cuda/tp.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/kernel_select.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_gptq_1.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_gptq_2.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_gptq_3.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_1a.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_1b.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_2a.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_2b.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_3a.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_3b.cu",
                    "exllamav2/exllamav2_ext/cpp/quantize_func.cpp",
                    "exllamav2/exllamav2_ext/cpp/profiling.cpp",
                    "exllamav2/exllamav2_ext/cpp/generator.cpp",
                    "exllamav2/exllamav2_ext/cpp/sampling.cpp",
                    "exllamav2/exllamav2_ext/cpp/sampling_avx2.cpp",
                ],
                extra_compile_args=extra_compile_args,
                libraries=["cublas"] if windows else [],
            )
        ],
        "cmdclass": {"build_ext": cpp_extension.BuildExtension},
    }
    if precompile and torch
    else {}
)

version_py = {}
with open("exllamav2/version.py", encoding="utf8") as fp:
    exec(fp.read(), version_py)
version = version_py["__version__"]
print("Version:", version)

# version = "0.0.5"

setup(
    name="exllamav2",
    version=version,
    packages=[
        "exllamav2",
        "exllamav2.generator",
        # "exllamav2.generator.filters",
        # "exllamav2.server",
        # "exllamav2.exllamav2_ext",
        # "exllamav2.exllamav2_ext.cpp",
        # "exllamav2.exllamav2_ext.cuda",
        # "exllamav2.exllamav2_ext.cuda.quant",
    ],
    url="https://github.com/turboderp/exllamav2",
    license="MIT",
    author="turboderp",
    install_requires=[
        "pandas",
        "ninja",
        "fastparquet",
        "torch>=2.2.0",
        "safetensors>=0.3.2",
        "pygments",
        "websockets",
        "regex",
        "numpy",
        "rich",
        "pillow>=9.1.0"
    ],
    include_package_data=True,
    package_data={
        "": ["py.typed"],
    },
    verbose=verbose,
    **setup_kwargs,
)
