"""Install the six patched vLLM 0.23 files that the serving tier requires.

vLLM validates `kv_cache_dtype` against a static allowlist and sizes the paged-KV pool from
per-token page bytes, so the packed SSNF/MX cache dtypes (`ssnf_u3gs16`, `mxint8_kv`, `mxfp8_kv`,
`mxfp6_kv`) need small in-tree edits: the dtype allowlist, the uint8 storage mapping, CUSTOM
attention-backend priority, and the packed-slot page-size accounting (reads `SSNF_KV_ASYM`).
Run AFTER `pip install -r env/requirements_serving.txt`:

    python env/apply_vllm_patches.py          # backs up each original to <file>.orig once
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys

FILES: dict[str, str] = {
    "config__cache.py": "config/cache.py",
    "utils__torch_utils.py": "utils/torch_utils.py",
    "platforms__cuda.py": "platforms/cuda.py",
    "platforms__interface.py": "platforms/interface.py",
    "model_executor__layers__attention__attention.py": "model_executor/layers/attention/attention.py",
    "v1__kv_cache_interface.py": "v1/kv_cache_interface.py",
}


def main() -> None:
    """Copy the patched files over the installed vLLM (version-guarded, originals kept as .orig)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("vllm is not installed in this environment — install requirements_serving.txt first")
    root = spec.submodule_search_locations[0]
    import vllm  # noqa: E402
    if not vllm.__version__.startswith("0.23"):
        sys.exit(f"these patches target vLLM 0.23.x; found {vllm.__version__} — aborting")
    here = os.path.dirname(os.path.abspath(__file__))
    for src, rel in FILES.items():
        dst = os.path.join(root, rel)
        if not os.path.exists(dst):
            sys.exit(f"unexpected vLLM layout: {rel} not found — aborting before any copy")
        if not os.path.exists(dst + ".orig"):
            shutil.copy2(dst, dst + ".orig")
        shutil.copy2(os.path.join(here, "vllm_patches", src), dst)
        print(f"patched {rel}")
    print("done — restore any file from its .orig copy to undo")


if __name__ == "__main__":
    main()
