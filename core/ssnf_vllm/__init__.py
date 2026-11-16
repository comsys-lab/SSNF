# Import alias: the code imports this package as `ssnf_vllm`; in this
# package layout the modules live in vllm_integration/ (and quant/ for the
# GPU packers and calibration helpers). Redirecting __path__ keeps the sources
# byte-identical to the working tree.
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
__path__ = [os.path.join(_root, "vllm_integration"), os.path.join(_root, "quant")]

# Importing this package registers the SSNF weight-quantization method ("ssnf")
# and the paged SSNF KV attention backend — IF vLLM is importable.
try:
    from ssnf_vllm import quant  # noqa: F401  — runs register_quantization_config("ssnf")
    from ssnf_vllm import quant_mxint8  # noqa: F401  — register_quantization_config("mxint8") (fair baseline)
    from ssnf_vllm import quant_mxfp8  # noqa: F401  — register_quantization_config("mxfp8") (native-FP twin)
    from ssnf_vllm import quant_mxfp6  # noqa: F401  — register_quantization_config("mxfp6w") (6.25b W-only)
    from ssnf_vllm import quant_fp4  # noqa: F401  — register "mxfp4"/"nvfp4" weight fake-quant (foils)
    from ssnf_vllm import kv_backend  # noqa: F401
    kv_backend.register()             # register SSNF paged-KV attention backend (CUSTOM)
except ImportError:               # vLLM or ssnf_cuda not importable (accuracy-only env, or an
                                  # extension built against a different torch) — registration skipped
    quant = None
    kv_backend = None
