#!/usr/bin/env python3
# setup.py  —  build the SSNF-signed CUDA/CUTLASS kernels into one extension.
#
#   python setup.py build_ext --inplace      # produces ssnf_cuda*.so next to this
#
# Produces a single `ssnf_cuda` module; importing it registers torch.ops.ssnf.*
# (see kernel/pybind.cpp). ssnf_lib.ops imports it lazily.
#
# Target: RTX PRO 4000 Blackwell = sm_120 (matches the gencode below and README §2).
#
# Optional environment toggles:
#   CUTLASS_DIR=/path/to/cutlass   add CUTLASS headers to the include path (only
#                                  needed once wa_gemm.cu adopts the CUTLASS
#                                  tensor-core path; the current baselines build
#                                  WITHOUT it).
#   ENABLE_PROFILING=1             compile in the clock64() micro-profiling spans
#                                  (ssnf_utils.cuh). Off by default — it perturbs
#                                  scheduling; use cuda.Event + ncu for real ms.

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_DIR = os.path.join(HERE, "kernel")

include_dirs = [KERNEL_DIR, os.path.join(KERNEL_DIR, "core")]
cutlass_dir = os.environ.get("CUTLASS_DIR")
if cutlass_dir:
    include_dirs.append(os.path.join(cutlass_dir, "include"))

nvcc_flags = ["-O3", "-gencode", "arch=compute_120,code=sm_120", "--use_fast_math",
              "--expt-extended-lambda"]   # device lambdas for the (u,gs)-specialized unpack helper
cxx_flags = ["-O3"]
if os.environ.get("ENABLE_PROFILING") == "1":
    nvcc_flags.append("-DENABLE_PROFILING")
    cxx_flags.append("-DENABLE_PROFILING")

setup(
    name="ssnf_cuda",
    ext_modules=[
        CUDAExtension(
            name="ssnf_cuda",
            sources=[
                os.path.join(KERNEL_DIR, "pybind.cpp"),
                os.path.join(KERNEL_DIR, "wa_gemm.cu"),
                os.path.join(KERNEL_DIR, "kv_attention.cu"),
                os.path.join(KERNEL_DIR, "kv_paged.cu"),
                os.path.join(KERNEL_DIR, "mxint8.cu"),
                os.path.join(KERNEL_DIR, "mxfp.cu"),
            ],
            include_dirs=include_dirs,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
