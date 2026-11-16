#!/usr/bin/env python3
"""Explicit, isolated extension build or read-only environment preflight."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from common import BUILD, HERE, REPO, CORE, file_hash, source_hashes, write_json


def build():
    import torch
    include = Path(torch.__file__).parent / "include"
    # Known local include-path corruption. This alias is local to this build;
    # neither installed torch/vLLM nor the core sources are edited.
    overlay = BUILD / "include/torch/msaq_runtime"
    if any("torch/msaq_runtime/csrc/" in p.read_text(errors="replace")
           for p in (include / "torch/csrc/api/include/torch").glob("*.h")):
        overlay.mkdir(parents=True, exist_ok=True)
        link = overlay / "csrc"
        if not link.exists():
            link.symlink_to(include / "torch/csrc", target_is_directory=True)
    env = os.environ.copy()
    env["CPATH"] = str(BUILD / "include") + (":"+env["CPATH"] if env.get("CPATH") else "")
    env.setdefault("MAX_JOBS", "2")
    if env.get("ENABLE_PROFILING") == "1":
        raise RuntimeError("Do not build instrumented kernels for energy measurement")
    subprocess.run([sys.executable, str(CORE / "setup.py"), "build_ext",
                    "--build-lib", str(BUILD / "lib"), "--build-temp", str(BUILD / "temp")],
                   cwd=REPO, env=env, check=True)
    binary = next((BUILD / "lib").glob("ssnf_cuda*.so"))
    write_json(BUILD / "manifest.json", dict(torch=torch.__version__, python=sys.version,
               binary=str(binary), binary_sha256=file_hash(binary), sources=source_hashes()))


def preflight(uuid):
    from telemetry import Telemetry
    meter = Telemetry(uuid)
    meter.assert_exclusive()
    result = dict(device=meter.metadata(), counter=meter.counter())
    meter.sample()
    result["power_sample"] = meter.samples[-1]
    # Lazy imports: --help and CPU-only unit tests never initialize CUDA.
    import torch
    import vllm
    if not vllm.__version__.startswith("0.23."):
        raise RuntimeError(f"This adapter targets vLLM 0.23, found {vllm.__version__}")
    result.update(torch=torch.__version__, vllm=vllm.__version__, python=sys.version)
    root = Path(importlib.util.find_spec("vllm").submodule_search_locations[0])
    from importlib.machinery import EXTENSION_SUFFIXES
    candidates = [BUILD / "lib" / ("ssnf_cuda"+suffix) for suffix in EXTENSION_SUFFIXES]
    result["local_extension_present"] = any(p.exists() for p in candidates)
    result["local_extension_manifest_valid"] = False
    if (BUILD / "manifest.json").exists():
        manifest = json.loads((BUILD / "manifest.json").read_text())
        binary = Path(manifest["binary"])
        result["local_extension_manifest_valid"] = (
            binary.exists() and file_hash(binary) == manifest["binary_sha256"]
            and torch.__version__ == manifest["torch"])
    cache = (root / "config/cache.py").read_text()
    result["packed_dtype_allowlist_present"] = all(x in cache for x in ("ssnf_u3gs16", "mxfp6_kv"))
    result["vllm_patch_hashes"] = {
        rel: file_hash(root / rel) for rel in (
            "config/cache.py", "utils/torch_utils.py", "platforms/cuda.py",
            "platforms/interface.py", "model_executor/layers/attention/attention.py",
            "v1/kv_cache_interface.py")}
    result["ready_for_smoke"] = (result["local_extension_present"] and
        result["local_extension_manifest_valid"] and result["packed_dtype_allowlist_present"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("build", "check"))
    p.add_argument("--gpu-uuid")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.action == "build":
        build()
    else:
        if not args.gpu_uuid:
            p.error("check requires --gpu-uuid")
        import json
        result = preflight(args.gpu_uuid)
        print(json.dumps(result, indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.out, result)
        if not result["ready_for_smoke"]:
            raise SystemExit("Preflight incomplete; see local extension and vLLM patch checks")


if __name__ == "__main__":
    main()
